import json
import sys
import subprocess
import requests
import time
from typing import Dict, Any, Optional, List
from loguru import logger


class MCPClient:
    """
    Minimal MCP client compatible with dojo360-ai-mcp-server
    """

    def __init__(
        self,
        transport_mode: str = "stdio",
        host: str = "127.0.0.1",
        port: int = 8080,
        url: str = None,
    ):
        self.transport_mode = transport_mode
        self.host = host
        self.port = port
        self.url = url  # full URL for deployed servers

        self.session_id = f"client-{int(time.time())}"
        self.request_id = 0
        self.process = None
        self.initialized = False

    def get_next_id(self) -> int:
        self.request_id += 1
        return self.request_id

    # ---------------- STDIO TRANSPORT ---------------- #

    def _send_stdio_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send request via stdio transport
        """
        if not self.process:
            raise RuntimeError("stdio process not started")

        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json)
        self.process.stdin.flush()

        response_line = self.process.stdout.readline().strip()
        return json.loads(response_line)

    # ---------------- HTTP TRANSPORT ---------------- #

    def _send_http_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send request via HTTP transport
        """
        if self.url:
            url = self.url
        else:
            url = f"http://{self.host}:{self.port}/api/mcp"

        headers = {
            "Content-Type": "application/json",
            "X-Session-ID": self.session_id,
        }

        logger.debug(f"Sending request to {url}: {request}")
        response = requests.post(
            url, json=request, headers=headers, timeout=300
        )

        logger.debug(f"Response status: {response.status_code}")
        logger.debug(f"Response text: {response.text[:200]}")

        response.raise_for_status()

        if not response.text.strip():
            raise ValueError("Empty response from server")

        return response.json()

    # ---------------- REQUEST DISPATCH ---------------- #

    def send_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send MCP request
        """
        request = {
            "jsonrpc": "2.0",
            "id": self.get_next_id(),
            "method": method,
        }

        if params:
            request["params"] = params

        try:
            if self.transport_mode == "stdio":
                return self._send_stdio_request(request)
            else:
                return self._send_http_request(request)
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    # ---------------- CONNECTION ---------------- #

    def connect(self) -> bool:
        """
        Establish connection with MCP server
        """
        try:
            if self.transport_mode == "stdio":
                self.process = subprocess.Popen(
                    [sys.executable, "main.py", "stdio"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=0,
                )

            # Initialize MCP
            response = self.send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "mcp-client",
                        "version": "1.0.0",
                    },
                },
            )

            if "error" in response:
                logger.error(f"Initialize failed: {response['error']}")
                return False

            self.initialized = True
            logger.info("MCP client connected successfully")
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """
        Close connection
        """
        if self.process:
            self.process.terminate()
            self.process = None
        self.initialized = False

    # ---------------- TOOLS ---------------- #

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Get available tools
        """
        if not self.initialized:
            raise RuntimeError("Client not initialized")

        response = self.send_request("tools/list")

        if "error" in response:
            raise RuntimeError(f"Failed to list tools: {response['error']}")

        return response["result"]["tools"]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool
        """
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

        # Check if this is an elicitation response
        if "elicitation" in content_text:
            return {
                "elicitation": True,
                "message": content_text,
            }

        return content_text

    def call_tool_interactive(
        self, name: str, arguments: Dict[str, Any]
    ) -> Any:
        """
        Call a tool with interactive elicitation handling
        """
        while True:
            result = self.call_tool(name, arguments)

            if isinstance(result, dict) and result.get("elicitation"):
                print("\n" + result["message"])
                print("Please provide the missing information:")

                user_input = input("Enter JSON with required fields: ")

                try:
                    additional_args = json.loads(user_input)
                    arguments.update(additional_args)
                except json.JSONDecodeError:
                    print("Invalid JSON format. Please try again.")
                    continue
            else:
                return result

    # ---------------- RESOURCES ---------------- #

    def list_resources(self) -> List[Dict[str, Any]]:
        """
        Get available resources
        """
        if not self.initialized:
            raise RuntimeError("Client not initialized")

        response = self.send_request("resources/list")

        if "error" in response:
            raise RuntimeError(
                f"Failed to list resources: {response['error']}"
            )

        return response["result"]["resources"]

    def read_resource(self, uri: str) -> Any:
        """
        Read a resource
        """
        if not self.initialized:
            raise RuntimeError("Client not initialized")

        response = self.send_request(
            "resources/read",
            {"uri": uri},
        )

        if "error" in response:
            raise RuntimeError(
                f"Failed to read resource: {response['error']}"
            )

        return response["result"]
