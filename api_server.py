"""
FastAPI server to expose the MCP client as a REST API endpoint.
"""
import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

from mcp_client.sse_client import MCPSSEClient
from mcp_client.streamable_http_client import MCPStreamableHTTPClient
from mcp_client.redis_client import get_redis_client

# Type alias for MCP clients
MCPClientType = MCPSSEClient | MCPStreamableHTTPClient

# Load environment variables
load_dotenv()


# Configuration
MCP_BASE_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
MCP_SERVERS_CONFIG = os.getenv("MCP_SERVERS", None)  # JSON array of server configs
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "8001"))


# ------------ MCP Server Manager ------------ #

class MCPServerManager:
    """
    Manages multiple MCP server connections with session persistence.
    
    Supports:
    - SSE MCP servers (GET /sse + POST /messages) - client_type: "sse"
    - Streamable HTTP MCP servers (POST /mcp) - client_type: "streamable_http"
    - Stateful servers (like Playwright) that need session persistence
    """
    
    def __init__(self):
        self.servers: Dict[str, MCPClientType] = {}
        self.server_configs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def _load_config_from_env(self):
        """Load server configurations from environment variables."""
        if MCP_SERVERS_CONFIG:
            try:
                configs = json.loads(MCP_SERVERS_CONFIG)
                for config in configs:
                    server_id = config.get("id", str(uuid.uuid4())[:8])
                    self.server_configs[server_id] = {
                        "id": server_id,
                        "url": config.get("url"),
                        "name": config.get("name", server_id),
                        "stateful": config.get("stateful", False),  # For Playwright-like servers
                        "client_type": config.get("client_type", "sse"),  # "sse" or "streamable_http"
                    }
                print(f"📋 Loaded {len(self.server_configs)} server configs from MCP_SERVERS")
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse MCP_SERVERS: {e}")
        
        # Backward compatibility: add MCP_URL as primary server if no configs
        if not self.server_configs and MCP_BASE_URL:
            self.server_configs["primary"] = {
                "id": "primary",
                "url": MCP_BASE_URL,
                "name": "Primary MCP Server",
                "stateful": False,
                "client_type": "sse",  # Default to SSE for backward compatibility
            }
    
    def add_server(
        self, 
        server_id: str, 
        url: str, 
        name: str = None, 
        stateful: bool = False,
        client_type: str = "sse"
    ) -> Dict[str, Any]:
        """
        Add a new server configuration and optionally connect.
        
        Args:
            server_id: Unique identifier for the server
            url: Base URL of the MCP server
            name: Human-readable name (defaults to server_id)
            stateful: Whether the server needs session persistence
            client_type: "sse" for standard SSE transport, "streamable_http" for Streamable HTTP
        """
        with self._lock:
            if server_id in self.server_configs:
                raise ValueError(f"Server '{server_id}' already exists")
            
            if client_type not in ("sse", "streamable_http"):
                raise ValueError(f"Invalid client_type: {client_type}. Must be 'sse' or 'streamable_http'")
            
            self.server_configs[server_id] = {
                "id": server_id,
                "url": url,
                "name": name or server_id,
                "stateful": stateful,
                "client_type": client_type,
            }
            
            return self.server_configs[server_id]
    
    def remove_server(self, server_id: str) -> bool:
        """Remove a server and disconnect if connected."""
        with self._lock:
            if server_id not in self.server_configs:
                return False
            
            # Disconnect if connected
            if server_id in self.servers:
                try:
                    self.servers[server_id].disconnect()
                except Exception:
                    pass
                del self.servers[server_id]
            
            del self.server_configs[server_id]
            return True
    
    def get_server(self, server_id: str) -> MCPClientType:
        """
        Get an MCP client for a server, connecting if necessary.
        
        Creates the appropriate client type based on server configuration:
        - "sse": MCPSSEClient (GET /sse + POST /messages)
        - "streamable_http": MCPStreamableHTTPClient (POST /mcp)
        
        Maintains session persistence for stateful servers.
        """
        with self._lock:
            if server_id not in self.server_configs:
                raise ValueError(f"Server '{server_id}' not found")
            
            config = self.server_configs[server_id]
            
            # Check if already connected and session is valid
            if server_id in self.servers:
                client = self.servers[server_id]
                if client.initialized:
                    return client
                # Session lost, need to reconnect
                print(f"🔄 Reconnecting to {server_id} (session expired)")
            
            # Create new connection based on client type
            client_type = config.get("client_type", "sse")
            
            if client_type == "streamable_http":
                client = MCPStreamableHTTPClient(base_url=config["url"])
            else:
                client = MCPSSEClient(base_url=config["url"])
            
            if client.connect():
                self.servers[server_id] = client
                print(f"✅ Connected to {config['name']} ({server_id}) [{client_type}]")
                return client
            else:
                raise ConnectionError(f"Failed to connect to server '{server_id}' at {config['url']}")
    
    def get_all_servers_status(self) -> List[Dict[str, Any]]:
        """Get status of all configured servers."""
        servers = []
        for server_id, config in self.server_configs.items():
            status = {
                "id": server_id,
                "name": config.get("name", server_id),
                "url": config.get("url"),
                "stateful": config.get("stateful", False),
                "client_type": config.get("client_type", "sse"),
                "status": "disconnected",
                "session_id": None,
                "capabilities": {"tools": 0, "resources": 0}
            }
            
            if server_id in self.servers:
                client = self.servers[server_id]
                if client.initialized:
                    status["status"] = "connected"
                    status["session_id"] = client.session_id
                    try:
                        status["capabilities"]["tools"] = len(client.list_tools())
                    except Exception:
                        pass
                    try:
                        status["capabilities"]["resources"] = len(client.list_resources())
                    except Exception:
                        pass
            
            servers.append(status)
        
        return servers
    
    def get_all_tools(self, server_id: str = None) -> List[Dict[str, Any]]:
        """Get tools from all servers or a specific server, including server_id in each tool."""
        all_tools = []
        
        target_servers = [server_id] if server_id else list(self.server_configs.keys())
        
        for sid in target_servers:
            try:
                client = self.get_server(sid)
                tools = client.list_tools()
                for tool in tools:
                    tool["server_id"] = sid
                    tool["server_name"] = self.server_configs[sid].get("name", sid)
                all_tools.extend(tools)
            except Exception as e:
                print(f"⚠️ Failed to get tools from {sid}: {e}")
        
        return all_tools
    
    def get_all_resources(self, server_id: str = None) -> List[Dict[str, Any]]:
        """Get resources from all servers or a specific server."""
        all_resources = []
        
        target_servers = [server_id] if server_id else list(self.server_configs.keys())
        
        for sid in target_servers:
            try:
                client = self.get_server(sid)
                resources = client.list_resources()
                for resource in resources:
                    resource["server_id"] = sid
                    resource["server_name"] = self.server_configs[sid].get("name", sid)
                all_resources.extend(resources)
            except Exception as e:
                print(f"⚠️ Failed to get resources from {sid}: {e}")
        
        return all_resources
    
    def call_tool(self, server_id: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on a specific server, maintaining session for stateful servers."""
        client = self.get_server(server_id)
        return client.call_tool(tool_name, arguments)
    
    def connect_all(self):
        """Connect to all configured servers."""
        for server_id in self.server_configs:
            try:
                self.get_server(server_id)
            except Exception as e:
                print(f"⚠️ Failed to connect to {server_id}: {e}")
    
    def disconnect_all(self):
        """Disconnect from all servers."""
        for server_id, client in list(self.servers.items()):
            try:
                client.disconnect()
                print(f"🔌 Disconnected from {server_id}")
            except Exception:
                pass
        self.servers.clear()


# Import threading for lock
import threading

# Global server manager instance
server_manager: Optional[MCPServerManager] = None

# Legacy global for backward compatibility
mcp_client: Optional[MCPSSEClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage MCP server connections on startup/shutdown."""
    global server_manager, mcp_client
    
    print(f"🚀 Starting API server...")
    
    # Initialize server manager
    server_manager = MCPServerManager()
    server_manager._load_config_from_env()
    
    print(f"📋 Configured servers: {list(server_manager.server_configs.keys())}")
    
    # Try to connect to all configured servers
    server_manager.connect_all()
    
    # Set legacy mcp_client for backward compatibility
    if "primary" in server_manager.servers:
        mcp_client = server_manager.servers["primary"]
    elif server_manager.servers:
        mcp_client = next(iter(server_manager.servers.values()))
    
    yield
    
    # Cleanup
    server_manager.disconnect_all()
    print("🔌 Disconnected from all MCP servers")



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
    server_id: Optional[str] = None  # Target server, uses first available if not specified


class AddServerRequest(BaseModel):
    """Request model for adding a new MCP server."""
    id: str
    url: str
    name: Optional[str] = None
    stateful: bool = False  # Set True for servers like Playwright that need session persistence
    client_type: str = "sse"  # "sse" for standard SSE transport, "streamable_http" for HTTP POST /mcp


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

def get_server_manager() -> MCPServerManager:
    """Get the server manager instance."""
    global server_manager
    if server_manager is None:
        server_manager = MCPServerManager()
        server_manager._load_config_from_env()
    return server_manager


def get_mcp_client(server_id: str = None) -> MCPSSEClient:
    """
    Get an MCP client instance for a specific server or the primary server.
    Maintains backward compatibility with single-server usage.
    """
    global mcp_client
    
    manager = get_server_manager()
    
    if server_id:
        # Get specific server
        try:
            return manager.get_server(server_id)
        except ValueError:
            raise HTTPException(status_code=404, detail=f"Server '{server_id}' not found")
        except ConnectionError as e:
            raise HTTPException(status_code=503, detail=str(e))
    
    # Use legacy mcp_client or first available server
    if mcp_client and mcp_client.initialized:
        return mcp_client
    
    # Try to get primary or first available
    if manager.server_configs:
        first_id = "primary" if "primary" in manager.server_configs else next(iter(manager.server_configs.keys()))
        try:
            client = manager.get_server(first_id)
            mcp_client = client  # Update legacy reference
            return client
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"No MCP server available: {e}")
    
    raise HTTPException(status_code=503, detail="No MCP servers configured")


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


@app.get("/health", include_in_schema=False)
@app.head("/health", include_in_schema=False)
async def health_check():
    """Minimal health check for Render/uptime monitoring. Supports both GET and HEAD. No side effects."""
    return Response(status_code=200)


@app.get("/internal/status")
async def internal_status():
    """Internal status endpoint with MCP connection details. NOT for health checks."""
    try:
        manager = get_server_manager()
        servers = manager.get_all_servers_status()
        
        # Check if any server is connected
        connected_servers = [s for s in servers if s["status"] == "connected"]
        
        return {
            "status": "ok",
            "servers": servers,
            "connected_count": len(connected_servers),
            "total_count": len(servers)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "servers": []
        }


@app.get("/servers")
async def list_servers():
    """List all available MCP servers and their status."""
    manager = get_server_manager()
    servers = manager.get_all_servers_status()
    return {
        "count": len(servers),
        "servers": servers
    }


@app.post("/servers")
async def add_server(request: AddServerRequest):
    """
    Add a new MCP server at runtime.
    
    Args:
        id: Unique server identifier
        url: Base URL of the MCP server
        name: Display name (optional)
        stateful: Set True for servers like Playwright that need session persistence
        client_type: "sse" for standard SSE transport, "streamable_http" for HTTP POST /mcp
    """
    manager = get_server_manager()
    
    try:
        server = manager.add_server(
            server_id=request.id,
            url=request.url,
            name=request.name,
            stateful=request.stateful,
            client_type=request.client_type
        )
        
        # Try to connect immediately
        try:
            manager.get_server(request.id)
            server["status"] = "connected"
        except Exception as e:
            server["status"] = "connection_failed"
            server["error"] = str(e)
        
        return {
            "success": True,
            "server": server
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/servers/{server_id}")
async def remove_server(server_id: str):
    """Remove an MCP server and disconnect if connected."""
    manager = get_server_manager()
    
    if manager.remove_server(server_id):
        return {"success": True, "message": f"Server '{server_id}' removed"}
    else:
        raise HTTPException(status_code=404, detail=f"Server '{server_id}' not found")


@app.get("/tools")
async def list_tools(server_id: Optional[str] = Query(None, description="Filter by server ID")):
    """
    List all available MCP tools.
    
    If server_id is provided, returns tools from that server only.
    Otherwise returns tools from all connected servers.
    """
    manager = get_server_manager()
    tools = manager.get_all_tools(server_id)
    return {
        "count": len(tools),
        "tools": tools
    }


@app.get("/tools/mistral")
async def list_tools_mistral_format(server_id: Optional[str] = Query(None)):
    """List all MCP tools in Mistral function-calling format."""
    manager = get_server_manager()
    tools = manager.get_all_tools(server_id)
    mistral_tools = convert_mcp_tools_to_mistral(tools)
    return {
        "count": len(mistral_tools),
        "tools": mistral_tools
    }


@app.post("/tools/call")
async def call_tool(request: ToolCallRequest):
    """
    Call a specific MCP tool.
    
    If server_id is provided, routes to that server.
    Otherwise uses primary/first available server.
    """
    manager = get_server_manager()
    
    # Determine which server to use
    server_id = request.server_id
    if not server_id:
        # Find server with this tool
        for sid, config in manager.server_configs.items():
            try:
                client = manager.get_server(sid)
                tools = client.list_tools()
                if any(t.get("name") == request.name for t in tools):
                    server_id = sid
                    break
            except Exception:
                continue
    
    if not server_id:
        server_id = "primary" if "primary" in manager.server_configs else next(iter(manager.server_configs.keys()), None)
    
    if not server_id:
        raise HTTPException(status_code=503, detail="No MCP servers available")
    
    try:
        result = manager.call_tool(server_id, request.name, request.arguments)
        
        # Check if result includes visualization
        response = {
            "success": True,
            "server_id": server_id,
            "tool": request.name,
            "result": result
        }
        
        if isinstance(result, dict) and "visualization" in result:
            response["result"] = result.get("content")
            response["visualization"] = result.get("visualization")
        
        return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")


@app.post("/tools/call/stream")
async def call_tool_stream(request: ToolCallRequest):
    """Call a specific MCP tool with streaming response (SSE)."""
    import asyncio
    
    manager = get_server_manager()
    
    # Determine which server to use
    server_id = request.server_id
    if not server_id:
        server_id = "primary" if "primary" in manager.server_configs else next(iter(manager.server_configs.keys()), None)
    
    if not server_id:
        return StreamingResponse(
            iter([f"data: {json.dumps({'event': 'error', 'message': 'No MCP servers available'})}\n\n"]),
            media_type="text/event-stream"
        )
    
    async def generate():
        try:
            # Send start event
            yield f"data: {json.dumps({'event': 'start', 'tool': request.name, 'server_id': server_id, 'status': 'processing'})}\n\n"
            
            # Call the tool
            result = manager.call_tool(server_id, request.name, request.arguments)
            
            # Check if result includes visualization
            if isinstance(result, dict) and "visualization" in result:
                # Send visualization event
                yield f"data: {json.dumps({'event': 'visualization', 'visualization': result.get('visualization')})}\n\n"
                # Send content
                content = result.get("content")
                if content:
                    yield f"data: {json.dumps({'event': 'result', 'content': content})}\n\n"
            # Stream the result
            elif isinstance(result, str):
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
async def list_resources(server_id: Optional[str] = Query(None, description="Filter by server ID")):
    """
    List all available MCP resources.
    
    If server_id is provided, returns resources from that server only.
    """
    manager = get_server_manager()
    resources = manager.get_all_resources(server_id)
    return {
        "count": len(resources),
        "resources": resources
    }


@app.get("/resources/{uri:path}")
async def read_resource(uri: str, server_id: Optional[str] = Query(None)):
    """Read a specific MCP resource."""
    client = get_mcp_client(server_id)
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
    
    def chat(self, user_prompt: str, max_iterations: int = 10, messages_history: List[Dict] = None) -> tuple:
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
            }
        ]
        
        # Add message history if provided (last 3 messages for context)
        if messages_history:
            messages.extend(messages_history)
        
        # Add current user prompt
        messages.append({"role": "user", "content": user_prompt})
        
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
                    
                    # Check if result contains visualization
                    if isinstance(result, str):
                        try:
                            result_obj = json.loads(result)
                            if isinstance(result_obj, dict) and "visualization" in result_obj:
                                # Send visualization as separate event
                                events.append({
                                    "event": "visualization",
                                    "tool": tool_name,
                                    "visualization": result_obj["visualization"]
                                })
                                # Send content separately (no truncation)
                                events.append({
                                    "event": "tool_result",
                                    "tool": tool_name,
                                    "result": result_obj.get("content", "")
                                })
                            else:
                                # No visualization, send full result
                                events.append({
                                    "event": "tool_result",
                                    "tool": tool_name,
                                    "result": result
                                })
                        except json.JSONDecodeError:
                            # Not JSON, send as-is
                            events.append({
                                "event": "tool_result",
                                "tool": tool_name,
                                "result": result
                            })
                    else:
                        # Result is not a string, send as-is
                        events.append({
                            "event": "tool_result",
                            "tool": tool_name,
                            "result": result
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
        
        # Get last 3 messages for context if continuing existing chat
        messages_history = []
        if request.chat_id:
            recent_messages = manager.get_chat_messages(chat_id)
            # Get last 3 messages (excluding the current one we're about to add)
            for msg in recent_messages[-3:]:
                if msg.role in ["user", "assistant"]:
                    messages_history.append({
                        "role": msg.role,
                        "content": msg.content
                    })
        
        # Store user message
        now = datetime.utcnow().isoformat() + "Z"
        user_message = ChatMessage(
            role="user",
            content=request.prompt,
            timestamp=now
        )
        manager.add_message(chat_id, user_message)
        
        # Run agent chat with history
        response, events = agent.chat(request.prompt, request.max_iterations, messages_history=messages_history)
        
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
            # Get last 3 messages for context if continuing existing chat
            messages_history = []
            if request.chat_id:
                recent_messages = manager.get_chat_messages(chat_id)
                # Get last 3 messages (excluding the current one we're about to add)
                for msg in recent_messages[-3:]:
                    if msg.role in ["user", "assistant"]:
                        messages_history.append({
                            "role": msg.role,
                            "content": msg.content
                        })
            
            # Stream events as they happen
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant with access to Yahoo Finance tools. "
                        "When users ask about stocks, financial data, or market information, "
                        "use the available tools to fetch real data. Be conversational and helpful."
                    )
                }
            ]
            
            # Add message history for context
            if messages_history:
                messages.extend(messages_history)
            
            # Add current user prompt
            messages.append({"role": "user", "content": request.prompt})
            
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
                        
                        # Check if result contains visualization
                        if isinstance(result, str):
                            try:
                                result_obj = json.loads(result)
                                if isinstance(result_obj, dict) and "visualization" in result_obj:
                                    # Send visualization as separate event (no truncation)
                                    yield f"data: {json.dumps({'event': 'visualization', 'tool': tool_name, 'visualization': result_obj['visualization']})}\n\n"
                                    # Send content separately (no truncation)
                                    content = result_obj.get("content", "")
                                    yield f"data: {json.dumps({'event': 'tool_result', 'tool': tool_name, 'result': content})}\n\n"
                                else:
                                    # No visualization, send full result (no truncation)
                                    yield f"data: {json.dumps({'event': 'tool_result', 'tool': tool_name, 'result': result})}\n\n"
                            except json.JSONDecodeError:
                                # Not JSON, send as-is (no truncation)
                                yield f"data: {json.dumps({'event': 'tool_result', 'tool': tool_name, 'result': result})}\n\n"
                        else:
                            # Result is not a string, send as-is
                            yield f"data: {json.dumps({'event': 'tool_result', 'tool': tool_name, 'result': result})}\n\n"
                        
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
