"""
MCP Client with Streamable HTTP transport support.

This client is designed for MCP servers that use:
- POST /mcp for JSON-RPC 2.0 requests (non-streaming)
- POST /mcp/stream for SSE streaming responses
- X-Session-ID header for session tracking

Example: Yahoo Finance MCP Server
"""
import json
import time
import uuid
from typing import Dict, Any, Optional, List, Generator
from loguru import logger
import requests
import os
from dotenv import load_dotenv

load_dotenv()


class MCPStreamableHTTPClient:
    """
    MCP client for Streamable HTTP transport.
    
    Unlike the SSE client which establishes a persistent SSE connection,
    this client uses simple HTTP POST requests with session tracking via headers.
    
    Endpoints:
    - POST /mcp - JSON-RPC 2.0 endpoint for all MCP operations
    - POST /mcp/stream - SSE streaming endpoint for tool calls
    - GET /health - Health check
    
    Headers:
    - Content-Type: application/json
    - X-Session-ID: <session-id> (optional, defaults to auto-generated)
    """

    def __init__(
        self,
        base_url: str = os.getenv("MCP_SERVER_URL", "http://localhost:8080"),
        session_id: str = None,
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id or f"client-{uuid.uuid4().hex[:8]}"
        self.timeout = timeout
        self.request_id = 0
        self.initialized = False
        self.server_info: Optional[Dict[str, Any]] = None

    def get_next_id(self) -> int:
        self.request_id += 1
        return self.request_id

    def _get_headers(self) -> Dict[str, str]:
        """Get standard headers for requests."""
        return {
            "Content-Type": "application/json",
            "X-Session-ID": self.session_id,
        }

    def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send a JSON-RPC 2.0 request to the /mcp endpoint.
        
        Args:
            method: The JSON-RPC method name
            params: Optional parameters for the method
            
        Returns:
            The JSON-RPC response as a dictionary
        """
        url = f"{self.base_url}/mcp"
        
        request = {
            "jsonrpc": "2.0",
            "id": self.get_next_id(),
            "method": method,
        }
        
        if params:
            request["params"] = params

        logger.debug(f"Sending request to {url}: {method}")
        
        try:
            response = requests.post(
                url,
                json=request,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
            
            logger.debug(f"Response status: {response.status_code}")
            
            if response.status_code not in (200, 202):
                raise RuntimeError(f"Request failed with status {response.status_code}: {response.text}")
            
            result = response.json()
            
            if "error" in result:
                error = result["error"]
                raise RuntimeError(f"JSON-RPC error [{error.get('code')}]: {error.get('message')}")
            
            return result
            
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to {url} timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to {url}: {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response: {e}")

    def connect(self) -> bool:
        """
        Initialize connection with the MCP server.
        
        Unlike SSE transport, this doesn't establish a persistent connection.
        Instead, it sends the initialize request and verifies the server is ready.
        """
        try:
            # Check health first (optional)
            try:
                health_url = f"{self.base_url}/health"
                health_response = requests.get(health_url, timeout=10)
                if health_response.status_code == 200:
                    logger.info(f"Server health check passed: {health_response.json()}")
            except Exception as e:
                logger.debug(f"Health check skipped: {e}")
            
            # Send initialize request
            response = self._send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "mcp-streamable-http-client",
                        "version": "1.0.0",
                    },
                },
            )

            self.server_info = response.get("result", {})
            logger.info(f"Server capabilities: {self.server_info.get('capabilities', {})}")
            
            self.initialized = True
            logger.info(f"MCP Streamable HTTP client connected to {self.base_url}")
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """
        Close the connection.
        
        For Streamable HTTP, this just resets state since there's no persistent connection.
        """
        self.initialized = False
        self.server_info = None
        logger.info("MCP Streamable HTTP client disconnected")

    def list_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from the server."""
        if not self.initialized:
            raise RuntimeError("Client not initialized - call connect() first")

        response = self._send_request("tools/list")
        return response.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the MCP server (non-streaming).
        
        Args:
            name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            The tool result (parsed JSON if possible, otherwise raw text)
        """
        if not self.initialized:
            raise RuntimeError("Client not initialized - call connect() first")

        response = self._send_request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments,
            },
        )

        result = response.get("result", {})
        content = result.get("content", [])
        
        if content and len(content) > 0:
            text = content[0].get("text", "")
            # Try to parse as JSON
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        
        return result

    def call_tool_streaming(
        self, 
        name: str, 
        arguments: Dict[str, Any]
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Call a tool with streaming response via SSE.
        
        Args:
            name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Yields:
            SSE events as dictionaries with keys like:
            - type: "chunk", "reasoning", "complete", "error", "end"
            - content: The content for chunk events
            - analysis: The full result for complete events
            - agent: The tool/agent name
        """
        if not self.initialized:
            raise RuntimeError("Client not initialized - call connect() first")

        url = f"{self.base_url}/mcp/stream"
        
        request = {
            "jsonrpc": "2.0",
            "id": self.get_next_id(),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments,
            },
        }

        logger.debug(f"Starting streaming request to {url} for tool: {name}")
        
        try:
            response = requests.post(
                url,
                json=request,
                headers=self._get_headers(),
                stream=True,
                timeout=None,  # No timeout for streaming
            )
            
            if response.status_code not in (200, 202):
                raise RuntimeError(f"Streaming request failed: {response.status_code}")
            
            event_type = None
            event_data = []
            
            for line in response.iter_lines(decode_unicode=True):
                if line is None:
                    continue
                    
                line = line.strip() if line else ""
                
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_content = line[5:].strip()
                    if data_content:
                        event_data.append(data_content)
                elif line == "" and event_data:
                    # End of event - parse and yield
                    data_str = "\n".join(event_data)
                    
                    try:
                        data = json.loads(data_str)
                        # Add event type if not in data
                        if event_type and "type" not in data:
                            data["event_type"] = event_type
                        yield data
                        
                        # Check for end event
                        if event_type == "end" or data.get("type") == "end":
                            break
                            
                    except json.JSONDecodeError:
                        # Yield raw data if not JSON
                        yield {"type": event_type or "raw", "content": data_str}
                    
                    event_type = None
                    event_data = []
                    
        except requests.exceptions.ConnectionError as e:
            yield {"type": "error", "message": f"Connection error: {e}"}
        except Exception as e:
            yield {"type": "error", "message": str(e)}

    def list_resources(self) -> List[Dict[str, Any]]:
        """Get available resources from the server."""
        if not self.initialized:
            raise RuntimeError("Client not initialized - call connect() first")

        response = self._send_request("resources/list")
        return response.get("result", {}).get("resources", [])

    def read_resource(self, uri: str) -> Any:
        """Read a resource by URI."""
        if not self.initialized:
            raise RuntimeError("Client not initialized - call connect() first")

        response = self._send_request(
            "resources/read",
            {"uri": uri},
        )
        return response.get("result", {})

    def send_batch_request(self, requests_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Send a batch of JSON-RPC requests for concurrent execution.
        
        Args:
            requests_list: List of request objects with 'method' and optional 'params'
            
        Returns:
            List of responses (order may vary due to concurrent execution)
        """
        if not self.initialized:
            raise RuntimeError("Client not initialized - call connect() first")

        url = f"{self.base_url}/mcp"
        
        batch = []
        for req in requests_list:
            batch.append({
                "jsonrpc": "2.0",
                "id": self.get_next_id(),
                "method": req.get("method"),
                "params": req.get("params", {}),
            })

        logger.debug(f"Sending batch request with {len(batch)} items")
        
        response = requests.post(
            url,
            json=batch,
            headers=self._get_headers(),
            timeout=self.timeout,
        )
        
        if response.status_code not in (200, 202):
            raise RuntimeError(f"Batch request failed: {response.status_code}")
        
        return response.json()
