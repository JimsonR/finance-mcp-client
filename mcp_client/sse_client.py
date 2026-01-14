"""
MCP Client with SSE (Server-Sent Events) transport support.
"""
import json
import time
import uuid
import threading
import queue
from typing import Dict, Any, Optional, List
from loguru import logger
import requests
import os
from dotenv import load_dotenv
load_dotenv()
class MCPSSEClient:
    """
    MCP client compatible with SSE transport (used by mcp.run and similar servers).
    
    SSE transport uses:
    - GET /sse - to establish SSE connection and receive session_id
    - POST /messages/?session_id=xxx - to send JSON-RPC requests
    """

    def __init__(
        self,
        base_url: str = os.getenv("MCP_SERVER_URL", "https://yahoo-finance-mcp-yk54.onrender.com"),
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        self.session_id: Optional[str] = None
        self.request_id = 0
        self.initialized = False
        
        # Queue to receive SSE responses
        self.response_queue: queue.Queue = queue.Queue()
        self.pending_requests: Dict[int, queue.Queue] = {}
        
        # SSE connection thread
        self.sse_thread: Optional[threading.Thread] = None
        self.sse_running = False

    def get_next_id(self) -> int:
        self.request_id += 1
        return self.request_id

    def _start_sse_listener(self):
        """Start SSE listener thread."""
        self.sse_running = True
        self.sse_thread = threading.Thread(target=self._sse_listener, daemon=True)
        self.sse_thread.start()

    def _sse_listener(self):
        """Listen for SSE events from the server."""
        sse_url = f"{self.base_url}/sse"
        logger.info(f"Connecting to SSE endpoint: {sse_url}")
        
        try:
            with requests.get(sse_url, stream=True, timeout=None) as response:
                response.raise_for_status()
                logger.info("SSE connection established")
                
                event_type = None
                event_data = []
                
                for line in response.iter_lines(decode_unicode=True):
                    if not self.sse_running:
                        break
                    
                    if line is None:
                        continue
                    
                    line = line.strip() if line else ""
                    
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        event_data.append(line[5:].strip())
                    elif line == "" and event_data:
                        # End of event
                        data = "\n".join(event_data)
                        self._handle_sse_event(event_type, data)
                        event_type = None
                        event_data = []
                        
        except Exception as e:
            logger.error(f"SSE connection error: {e}")
            self.sse_running = False

    def _handle_sse_event(self, event_type: Optional[str], data: str):
        """Handle incoming SSE event."""
        logger.debug(f"SSE event: type={event_type}, data={data[:200]}")
        
        if event_type == "endpoint":
            # Server sends the message endpoint URL
            # Format: /messages/?session_id=xxx
            if "session_id=" in data:
                self.session_id = data.split("session_id=")[1].split("&")[0]
                logger.info(f"Received session_id: {self.session_id}")
            self.response_queue.put({"type": "endpoint", "data": data})
        elif event_type == "message":
            # JSON-RPC response
            try:
                response = json.loads(data)
                request_id = response.get("id")
                
                # Handle both int and string IDs
                if request_id is not None:
                    # Try both int and string versions of the ID
                    int_id = int(request_id) if isinstance(request_id, str) else request_id
                    str_id = str(request_id)
                    
                    logger.debug(f"Looking for pending request: {int_id} or {str_id}, pending: {list(self.pending_requests.keys())}")
                    
                    if int_id in self.pending_requests:
                        self.pending_requests[int_id].put(response)
                        logger.debug(f"Matched response to request {int_id}")
                    elif str_id in self.pending_requests:
                        self.pending_requests[str_id].put(response)
                        logger.debug(f"Matched response to request {str_id}")
                    else:
                        logger.warning(f"No pending request for ID {request_id}")
                        self.response_queue.put({"type": "message", "data": response})
                else:
                    self.response_queue.put({"type": "message", "data": response})
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse SSE message: {e}")
        else:
            logger.debug(f"Unknown SSE event type: {event_type}")

    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send JSON-RPC request via POST to messages endpoint."""
        if not self.session_id:
            raise RuntimeError("No session_id - SSE connection not established")
        
        messages_url = f"{self.base_url}/messages/?session_id={self.session_id}"
        
        headers = {
            "Content-Type": "application/json",
        }
        
        request_id = request.get("id")
        
        # Create a queue for this request's response
        response_queue = queue.Queue()
        self.pending_requests[request_id] = response_queue
        
        try:
            logger.debug(f"Sending request to {messages_url}: {request}")
            response = requests.post(
                messages_url, 
                json=request, 
                headers=headers, 
                timeout=self.timeout
            )
            
            logger.debug(f"POST response status: {response.status_code}")
            
            if response.status_code not in (200, 202):
                raise RuntimeError(f"Request failed with status {response.status_code}: {response.text}")
            
            # Wait for the response via SSE
            try:
                result = response_queue.get(timeout=self.timeout)
                return result
            except queue.Empty:
                raise TimeoutError(f"Timeout waiting for response to request {request_id}")
                
        finally:
            # Clean up pending request
            self.pending_requests.pop(request_id, None)

    def send_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send MCP request."""
        request = {
            "jsonrpc": "2.0",
            "id": self.get_next_id(),
            "method": method,
        }

        if params:
            request["params"] = params

        try:
            return self._send_request(request)
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    def connect(self) -> bool:
        """Establish connection with MCP server via SSE."""
        try:
            # Start SSE listener
            self._start_sse_listener()
            
            # Wait for session_id
            max_wait = 10  # seconds
            start_time = time.time()
            while not self.session_id and (time.time() - start_time) < max_wait:
                time.sleep(0.1)
            
            if not self.session_id:
                logger.error("Timeout waiting for session_id from SSE")
                return False
            
            logger.info(f"Got session_id: {self.session_id}")
            
            # Step 1: Send initialize request
            response = self.send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "mcp-sse-client",
                        "version": "1.0.0",
                    },
                },
            )

            if "error" in response:
                logger.error(f"Initialize failed: {response['error']}")
                return False
            
            logger.info("Initialize response received, sending initialized notification...")
            
            # Step 2: Send notifications/initialized to complete handshake
            # This is a notification (no response expected), so we send it differently
            self._send_notification("notifications/initialized", {})
            
            # Give the server a moment to process the notification
            time.sleep(0.2)

            self.initialized = True
            logger.info("MCP SSE client connected successfully")
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def _send_notification(self, method: str, params: Dict[str, Any] = None):
        """Send a notification (no response expected)."""
        if not self.session_id:
            raise RuntimeError("No session_id - SSE connection not established")
        
        messages_url = f"{self.base_url}/messages/?session_id={self.session_id}"
        
        notification = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            notification["params"] = params
        
        headers = {"Content-Type": "application/json"}
        
        logger.debug(f"Sending notification: {notification}")
        response = requests.post(messages_url, json=notification, headers=headers, timeout=30)
        logger.debug(f"Notification response status: {response.status_code}")

    def disconnect(self):
        """Close connection."""
        self.sse_running = False
        if self.sse_thread:
            self.sse_thread.join(timeout=2)
        self.initialized = False
        self.session_id = None

    def list_tools(self) -> List[Dict[str, Any]]:
        """Get available tools."""
        if not self.initialized:
            raise RuntimeError("Client not initialized")

        response = self.send_request("tools/list")

        if "error" in response:
            raise RuntimeError(f"Failed to list tools: {response['error']}")

        return response["result"]["tools"]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool."""
        if not self.initialized:
            raise RuntimeError("Client not initialized")

        response = self.send_request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments,
            },
        )

        if "error" in response:
            raise RuntimeError(f"Tool call failed: {response['error']}")

        content_text = response["result"]["content"][0]["text"]
        return content_text

    def list_resources(self) -> List[Dict[str, Any]]:
        """Get available resources."""
        if not self.initialized:
            raise RuntimeError("Client not initialized")

        response = self.send_request("resources/list")

        if "error" in response:
            raise RuntimeError(f"Failed to list resources: {response['error']}")

        return response["result"]["resources"]

    def read_resource(self, uri: str) -> Any:
        """Read a resource."""
        if not self.initialized:
            raise RuntimeError("Client not initialized")

        response = self.send_request(
            "resources/read",
            {"uri": uri},
        )

        if "error" in response:
            raise RuntimeError(f"Failed to read resource: {response['error']}")

        return response["result"]
