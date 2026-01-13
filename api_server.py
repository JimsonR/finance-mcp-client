"""
FastAPI server to expose the MCP client as a REST API endpoint.
"""
import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

from mcp_client.sse_client import MCPSSEClient
from mcp_client.redis_client import get_redis_client

# Load environment variables
load_dotenv()


# Configuration
MCP_BASE_URL = os.getenv("MCP_URL", "http://localhost:8000")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "8001"))

# Global MCP client instance
mcp_client: Optional[MCPSSEClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage MCP client connection on startup/shutdown."""
    global mcp_client
    
    print(f"🚀 Starting API server...")
    print(f"📡 MCP Server URL: {MCP_BASE_URL}")
    
    # Try to connect to MCP server, but don't fail if unavailable
    mcp_client = MCPSSEClient(base_url=MCP_BASE_URL)
    
    try:
        if mcp_client.connect():
            tools = mcp_client.list_tools()
            print(f"✅ Connected to MCP server. Found {len(tools)} tools.")
        else:
            print(f"⚠️ MCP server not available at {MCP_BASE_URL}. Will retry on first request.")
            mcp_client = None
    except Exception as e:
        print(f"⚠️ Could not connect to MCP server: {e}. Will retry on first request.")
        mcp_client = None
    
    yield
    
    # Cleanup
    if mcp_client:
        mcp_client.disconnect()
        print("🔌 Disconnected from MCP server")



# Initialize FastAPI app
app = FastAPI(
    title="MCP Client API",
    description="REST API to interact with the MCP server tools",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------ Request/Response Models ------------ #

class ToolCallRequest(BaseModel):
    """Request model for calling a tool."""
    name: str
    arguments: Dict[str, Any] = {}


class ChatRequest(BaseModel):
    """Request model for chat with agent."""
    prompt: str
    max_iterations: int = 10
    chat_id: Optional[str] = None  # Existing chat to continue


class StoreDataRequest(BaseModel):
    """Request model for storing data in memory."""
    variable_name: str
    data: Dict[str, Any]


class AnalysisRequest(BaseModel):
    """Request model for running analysis."""
    analysis_type: str  # debt, liquidity, qoe, asset
    data: Optional[Dict[str, Any]] = None
    variable_name: str = "blackboard_data"


# ------------ Chat Management Models ------------ #

class ChatSession(BaseModel):
    """Chat session metadata."""
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class ChatMessage(BaseModel):
    """Individual chat message."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: str
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


class CreateChatRequest(BaseModel):
    """Request model for creating a chat."""
    title: Optional[str] = None


class UpdateChatRequest(BaseModel):
    """Request model for updating a chat."""
    title: str


# ------------ Chat Manager (Redis) ------------ #

class ChatManager:
    """Manages chat sessions and messages in Redis."""
    
    CHAT_PREFIX = "chat:"
    CHATS_INDEX = "chats:index"
    
    def __init__(self):
        self.redis = get_redis_client()
    
    def _ensure_connection(self):
        """Ensure Redis connection is available."""
        if self.redis is None:
            self.redis = get_redis_client()
        if self.redis is None:
            raise HTTPException(status_code=503, detail="Redis not available")
        return self.redis
    
    def create_chat(self, title: Optional[str] = None) -> ChatSession:
        """Create a new chat session."""
        r = self._ensure_connection()
        
        chat_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat() + "Z"
        
        if not title:
            title = f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
        
        chat_data = {
            "id": chat_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "message_count": "0"
        }
        
        # Store chat metadata
        r.hset(f"{self.CHAT_PREFIX}{chat_id}", mapping=chat_data)
        # Add to index
        r.sadd(self.CHATS_INDEX, chat_id)
        
        return ChatSession(
            id=chat_id,
            title=title,
            created_at=now,
            updated_at=now,
            message_count=0
        )
    
    def list_chats(self) -> List[ChatSession]:
        """List all chat sessions."""
        r = self._ensure_connection()
        
        chat_ids = r.smembers(self.CHATS_INDEX)
        chats = []
        
        for chat_id in chat_ids:
            data = r.hgetall(f"{self.CHAT_PREFIX}{chat_id}")
            if data:
                chats.append(ChatSession(
                    id=data.get("id", chat_id),
                    title=data.get("title", "Untitled"),
                    created_at=data.get("created_at", ""),
                    updated_at=data.get("updated_at", ""),
                    message_count=int(data.get("message_count", 0))
                ))
        
        # Sort by updated_at descending
        chats.sort(key=lambda x: x.updated_at, reverse=True)
        return chats
    
    def get_chat(self, chat_id: str) -> Optional[ChatSession]:
        """Get chat session metadata."""
        r = self._ensure_connection()
        
        data = r.hgetall(f"{self.CHAT_PREFIX}{chat_id}")
        if not data:
            return None
        
        return ChatSession(
            id=data.get("id", chat_id),
            title=data.get("title", "Untitled"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            message_count=int(data.get("message_count", 0))
        )
    
    def get_chat_messages(self, chat_id: str) -> List[ChatMessage]:
        """Get all messages for a chat."""
        r = self._ensure_connection()
        
        messages_key = f"{self.CHAT_PREFIX}{chat_id}:messages"
        raw_messages = r.lrange(messages_key, 0, -1)
        
        messages = []
        for raw in raw_messages:
            try:
                msg_data = json.loads(raw)
                messages.append(ChatMessage(**msg_data))
            except (json.JSONDecodeError, ValueError):
                continue
        
        return messages
    
    def add_message(self, chat_id: str, message: ChatMessage) -> None:
        """Add a message to a chat."""
        r = self._ensure_connection()
        
        messages_key = f"{self.CHAT_PREFIX}{chat_id}:messages"
        r.rpush(messages_key, message.model_dump_json())
        
        # Update chat metadata
        now = datetime.utcnow().isoformat() + "Z"
        r.hset(f"{self.CHAT_PREFIX}{chat_id}", "updated_at", now)
        r.hincrby(f"{self.CHAT_PREFIX}{chat_id}", "message_count", 1)
    
    def update_chat(self, chat_id: str, title: str) -> Optional[ChatSession]:
        """Update chat title."""
        r = self._ensure_connection()
        
        chat_key = f"{self.CHAT_PREFIX}{chat_id}"
        if not r.exists(chat_key):
            return None
        
        now = datetime.utcnow().isoformat() + "Z"
        r.hset(chat_key, mapping={"title": title, "updated_at": now})
        
        return self.get_chat(chat_id)
    
    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat session and its messages."""
        r = self._ensure_connection()
        
        chat_key = f"{self.CHAT_PREFIX}{chat_id}"
        messages_key = f"{self.CHAT_PREFIX}{chat_id}:messages"
        
        if not r.exists(chat_key):
            return False
        
        # Delete chat data and messages
        r.delete(chat_key)
        r.delete(messages_key)
        r.srem(self.CHATS_INDEX, chat_id)
        
        return True


# Global chat manager instance
chat_manager: Optional[ChatManager] = None

def get_chat_manager() -> ChatManager:
    """Get or create the chat manager."""
    global chat_manager
    if chat_manager is None:
        chat_manager = ChatManager()
    return chat_manager


# ------------ Helper Functions ------------ #

def get_mcp_client() -> MCPSSEClient:
    """Get the MCP client instance, attempting to connect if not already connected."""
    global mcp_client
    
    if mcp_client is None:
        # Try to connect now
        mcp_client = MCPSSEClient(base_url=MCP_BASE_URL)
        if not mcp_client.connect():
            mcp_client = None
            raise HTTPException(
                status_code=503, 
                detail=f"MCP server not available at {MCP_BASE_URL}. Please ensure it's running."
            )
    
    if not mcp_client.initialized:
        raise HTTPException(status_code=503, detail="MCP client not initialized")
    
    return mcp_client


def convert_mcp_tools_to_mistral(mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert MCP tool definitions to Mistral function-calling format."""
    mistral_tools = []
    
    for tool in mcp_tools:
        name = tool.get("name")
        if not name:
            continue
            
        schema = tool.get("inputSchema") or tool.get("input_schema") or {
            "type": "object",
            "properties": {},
            "additionalProperties": True
        }
        
        mistral_tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description", ""),
                "parameters": schema
            }
        })
    
    return mistral_tools


# ------------ API Endpoints ------------ #

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "MCP Client API",
        "version": "1.0.0",
        "docs": "/docs",
        "mcp_url": MCP_BASE_URL
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    client = get_mcp_client()
    return {
        "status": "healthy",
        "mcp_connected": client.initialized,
        "session_id": client.session_id
    }


@app.get("/servers")
async def list_servers():
    """List all available MCP servers and their status."""
    servers = []
    
    # Primary MCP server
    server_info = {
        "name": "primary",
        "url": MCP_BASE_URL,
        "status": "unknown",
        "session_id": None,
        "capabilities": {
            "tools": 0,
            "resources": 0
        }
    }
    
    try:
        client = get_mcp_client()
        if client and client.initialized:
            server_info["status"] = "connected"
            server_info["session_id"] = client.session_id
            
            # Get capabilities
            try:
                tools = client.list_tools()
                server_info["capabilities"]["tools"] = len(tools)
            except Exception:
                pass
            
            try:
                resources = client.list_resources()
                server_info["capabilities"]["resources"] = len(resources)
            except Exception:
                pass
        else:
            server_info["status"] = "disconnected"
    except HTTPException:
        server_info["status"] = "unavailable"
    except Exception as e:
        server_info["status"] = "error"
        server_info["error"] = str(e)
    
    servers.append(server_info)
    
    return {
        "count": len(servers),
        "servers": servers
    }


@app.get("/tools")
async def list_tools():
    """List all available MCP tools."""
    client = get_mcp_client()
    tools = client.list_tools()
    return {
        "count": len(tools),
        "tools": tools
    }


@app.get("/tools/mistral")
async def list_tools_mistral_format():
    """List all MCP tools in Mistral function-calling format."""
    client = get_mcp_client()
    tools = client.list_tools()
    mistral_tools = convert_mcp_tools_to_mistral(tools)
    return {
        "count": len(mistral_tools),
        "tools": mistral_tools
    }


@app.post("/tools/call")
async def call_tool(request: ToolCallRequest):
    """Call a specific MCP tool."""
    client = get_mcp_client()
    
    try:
        result = client.call_tool(request.name, request.arguments)
        return {
            "success": True,
            "tool": request.name,
            "result": result
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")


@app.post("/tools/call/stream")
async def call_tool_stream(request: ToolCallRequest):
    """Call a specific MCP tool with streaming response (SSE)."""
    import asyncio
    
    client = get_mcp_client()
    
    async def generate():
        try:
            # Send start event
            yield f"data: {json.dumps({'event': 'start', 'tool': request.name, 'status': 'processing'})}\n\n"
            
            # Call the tool
            result = client.call_tool(request.name, request.arguments)
            
            # Stream the result
            if isinstance(result, str):
                # For large results, stream in chunks
                if len(result) > 1000:
                    chunk_size = 500
                    chunks = [result[i:i+chunk_size] for i in range(0, len(result), chunk_size)]
                    for i, chunk in enumerate(chunks):
                        yield f"data: {json.dumps({'event': 'chunk', 'index': i+1, 'total': len(chunks), 'content': chunk})}\n\n"
                        await asyncio.sleep(0.01)
                else:
                    yield f"data: {json.dumps({'event': 'result', 'content': result})}\n\n"
            else:
                yield f"data: {json.dumps({'event': 'result', 'content': result})}\n\n"
            
            # Send completion event
            yield f"data: {json.dumps({'event': 'complete', 'status': 'success'})}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/resources")
async def list_resources():
    """List all available MCP resources."""
    client = get_mcp_client()
    try:
        resources = client.list_resources()
        return {
            "count": len(resources),
            "resources": resources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/resources/{uri:path}")
async def read_resource(uri: str):
    """Read a specific MCP resource."""
    client = get_mcp_client()
    try:
        result = client.read_resource(uri)
        return {
            "uri": uri,
            "content": result
        }
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ------------ Chat Management Endpoints ------------ #

@app.post("/chats")
async def create_chat(request: CreateChatRequest = None):
    """Create a new chat session."""
    manager = get_chat_manager()
    title = request.title if request else None
    chat = manager.create_chat(title)
    return {
        "success": True,
        "chat": chat.model_dump()
    }


@app.get("/chats")
async def list_chats():
    """List all chat sessions."""
    manager = get_chat_manager()
    chats = manager.list_chats()
    return {
        "count": len(chats),
        "chats": [chat.model_dump() for chat in chats]
    }


@app.get("/chats/{chat_id}")
async def get_chat(chat_id: str, include_messages: bool = Query(True)):
    """Get a chat session with optional message history."""
    manager = get_chat_manager()
    chat = manager.get_chat(chat_id)
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    result = {"chat": chat.model_dump()}
    
    if include_messages:
        messages = manager.get_chat_messages(chat_id)
        result["messages"] = [msg.model_dump() for msg in messages]
    
    return result


@app.put("/chats/{chat_id}")
async def update_chat(chat_id: str, request: UpdateChatRequest):
    """Update a chat session (rename)."""
    manager = get_chat_manager()
    chat = manager.update_chat(chat_id, request.title)
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    return {
        "success": True,
        "chat": chat.model_dump()
    }


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat session and all its messages."""
    manager = get_chat_manager()
    deleted = manager.delete_chat(chat_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    return {"success": True, "message": f"Chat {chat_id} deleted"}

# ------------ Conversational Chat with LLM ------------ #

import requests as http_requests  # Rename to avoid conflict

class ConversationalAgent:
    """AI Agent that uses MCP tools via Mistral API for conversational interactions."""
    
    def __init__(self, mcp_client: MCPSSEClient, mistral_api_key: str, model: str = "mistral-large-latest"):
        self.mcp_client = mcp_client
        self.mistral_api_key = mistral_api_key
        self.model = model
        self.mistral_url = "https://api.mistral.ai/v1/chat/completions"
        self.tools = self._convert_mcp_tools_to_mistral()
    
    def _convert_mcp_tools_to_mistral(self) -> List[Dict[str, Any]]:
        """Convert MCP tool definitions to Mistral function-calling format."""
        mcp_tools = self.mcp_client.list_tools()
        mistral_tools = []
        
        for tool in mcp_tools:
            name = tool.get("name")
            if not name:
                continue
            
            schema = tool.get("inputSchema") or tool.get("input_schema") or {
                "type": "object",
                "properties": {},
                "additionalProperties": True
            }
            
            mistral_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": schema
                }
            })
        
        return mistral_tools
    
    def call_mistral(self, messages: List[Dict], tools: List = None, max_tokens: int = 4096) -> Dict:
        """Call Mistral API with messages and optional tools."""
        headers = {
            "Authorization": f"Bearer {self.mistral_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        response = http_requests.post(
            self.mistral_url,
            json=payload,
            headers=headers,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    
    def execute_tool(self, tool_name: str, arguments: Dict) -> str:
        """Execute an MCP tool and return the result."""
        available_tools = [t["function"]["name"] for t in self.tools]
        if tool_name not in available_tools:
            return json.dumps({"error": f"Tool '{tool_name}' not found"})
        
        try:
            result = self.mcp_client.call_tool(tool_name, arguments or {})
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def chat(self, user_prompt: str, max_iterations: int = 10) -> tuple:
        """
        Run conversational chat with tool calling.
        Returns (final_response, events_list) for streaming.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant with access to Yahoo Finance tools. "
                    "When users ask about stocks, financial data, or market information, "
                    "use the available tools to fetch real data. Be conversational and helpful. "
                    "After fetching data, provide a clear and insightful summary."
                )
            },
            {"role": "user", "content": user_prompt}
        ]
        
        events = []
        events.append({"event": "thinking", "message": "Processing your request..."})
        
        for iteration in range(max_iterations):
            response = self.call_mistral(messages, tools=self.tools)
            
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls") or []
            
            if tool_calls:
                events.append({
                    "event": "tool_calls", 
                    "count": len(tool_calls),
                    "tools": [tc["function"]["name"] for tc in tool_calls]
                })
                
                messages.append({"role": "assistant", "tool_calls": tool_calls})
                
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    tool_name = fn.get("name")
                    raw_args = fn.get("arguments", "{}")
                    
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = {}
                    
                    events.append({
                        "event": "executing_tool",
                        "tool": tool_name,
                        "arguments": args
                    })
                    
                    result = self.execute_tool(tool_name, args)
                    
                    events.append({
                        "event": "tool_result",
                        "tool": tool_name,
                        "result_preview": result[:200] + "..." if len(result) > 200 else result
                    })
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "content": result
                    })
                
                continue
            
            # No tool calls - final response
            content = message.get("content", "")
            if content:
                events.append({"event": "response", "content": content})
                return content, events
        
        events.append({"event": "max_iterations", "message": "Reached maximum iterations"})
        return "I couldn't complete your request within the allowed iterations.", events


# Global agent instance (initialized lazily)
agent_instance: Optional[ConversationalAgent] = None

def get_agent() -> ConversationalAgent:
    """Get or create the conversational agent."""
    global agent_instance
    if agent_instance is None:
        if not MISTRAL_API_KEY:
            raise HTTPException(status_code=500, detail="MISTRAL_API_KEY not configured")
        client = get_mcp_client()
        agent_instance = ConversationalAgent(client, MISTRAL_API_KEY)
    return agent_instance


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Conversational chat endpoint. Send a natural language prompt and get a response.
    The AI will automatically use MCP tools when needed.
    
    Optionally provide a chat_id to continue an existing conversation.
    Messages are persisted to Redis for history.
    """
    agent = get_agent()
    manager = get_chat_manager()
    
    try:
        # Get or create chat session
        chat_id = request.chat_id
        if chat_id:
            chat = manager.get_chat(chat_id)
            if not chat:
                raise HTTPException(status_code=404, detail="Chat not found")
        else:
            # Create new chat with first prompt as title (truncated)
            title = request.prompt[:50] + "..." if len(request.prompt) > 50 else request.prompt
            chat = manager.create_chat(title)
            chat_id = chat.id
        
        # Store user message
        now = datetime.utcnow().isoformat() + "Z"
        user_message = ChatMessage(
            role="user",
            content=request.prompt,
            timestamp=now
        )
        manager.add_message(chat_id, user_message)
        
        # Run agent chat
        response, events = agent.chat(request.prompt, request.max_iterations)
        
        # Store assistant response
        assistant_message = ChatMessage(
            role="assistant",
            content=response,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        manager.add_message(chat_id, assistant_message)
        
        return {
            "success": True,
            "chat_id": chat_id,
            "response": response,
            "events": events
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Conversational chat with streaming SSE response.
    Events are streamed as they happen (thinking, tool calls, results, final response).
    
    Optionally provide a chat_id to continue an existing conversation.
    Messages are persisted to Redis for history.
    """
    import asyncio
    
    agent = get_agent()
    manager = get_chat_manager()
    
    # Get or create chat session before streaming
    chat_id = request.chat_id
    if chat_id:
        chat = manager.get_chat(chat_id)
        if not chat:
            return StreamingResponse(
                iter([f"data: {json.dumps({'event': 'error', 'message': 'Chat not found'})}\n\n", "data: [DONE]\n\n"]),
                media_type="text/event-stream"
            )
    else:
        title = request.prompt[:50] + "..." if len(request.prompt) > 50 else request.prompt
        chat = manager.create_chat(title)
        chat_id = chat.id
    
    # Store user message immediately
    now = datetime.utcnow().isoformat() + "Z"
    user_message = ChatMessage(
        role="user",
        content=request.prompt,
        timestamp=now
    )
    manager.add_message(chat_id, user_message)
    
    async def generate():
        final_response = ""
        try:
            # Stream events as they happen
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant with access to Yahoo Finance tools. "
                        "When users ask about stocks, financial data, or market information, "
                        "use the available tools to fetch real data. Be conversational and helpful."
                    )
                },
                {"role": "user", "content": request.prompt}
            ]
            
            yield f"data: {json.dumps({'event': 'start', 'chat_id': chat_id, 'message': 'Processing your request...'})}\n\n"
            
            for iteration in range(request.max_iterations):
                yield f"data: {json.dumps({'event': 'thinking', 'iteration': iteration + 1})}\n\n"
                
                response = agent.call_mistral(messages, tools=agent.tools)
                
                choice = response.get("choices", [{}])[0]
                message = choice.get("message", {})
                tool_calls = message.get("tool_calls") or []
                
                if tool_calls:
                    yield f"data: {json.dumps({'event': 'tool_calls', 'count': len(tool_calls), 'tools': [tc['function']['name'] for tc in tool_calls]})}\n\n"
                    
                    messages.append({"role": "assistant", "tool_calls": tool_calls})
                    
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        tool_name = fn.get("name")
                        raw_args = fn.get("arguments", "{}")
                        
                        try:
                            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except json.JSONDecodeError:
                            args = {}
                        
                        yield f"data: {json.dumps({'event': 'executing_tool', 'tool': tool_name, 'arguments': args})}\n\n"
                        await asyncio.sleep(0.01)
                        
                        result = agent.execute_tool(tool_name, args)
                        
                        # Stream tool result preview
                        preview = result[:500] + "..." if len(result) > 500 else result
                        yield f"data: {json.dumps({'event': 'tool_result', 'tool': tool_name, 'result': preview})}\n\n"
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id"),
                            "content": result
                        })
                    
                    continue
                
                # Final response
                content = message.get("content", "")
                if content:
                    final_response = content
                    # Stream the response in chunks for effect
                    words = content.split()
                    chunk_size = 10
                    for i in range(0, len(words), chunk_size):
                        chunk = " ".join(words[i:i+chunk_size])
                        yield f"data: {json.dumps({'event': 'response_chunk', 'content': chunk})}\n\n"
                        await asyncio.sleep(0.02)
                    
                    yield f"data: {json.dumps({'event': 'complete', 'chat_id': chat_id, 'full_response': content})}\n\n"
                    break
            
            # Store assistant response after streaming completes
            if final_response:
                assistant_message = ChatMessage(
                    role="assistant",
                    content=final_response,
                    timestamp=datetime.utcnow().isoformat() + "Z"
                )
                manager.add_message(chat_id, assistant_message)
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    print(f"🚀 Starting MCP Client API server on {HOST}:{PORT}")
    print(f"📡 MCP Server URL: {MCP_BASE_URL}")
    print(f"📚 API docs available at http://{HOST}:{PORT}/docs")
    
    uvicorn.run(
        "api_server:app",
        host=HOST,
        port=PORT,
        reload=True
    )
