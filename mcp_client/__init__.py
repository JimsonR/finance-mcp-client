"""MCP Client Package"""
__version__ = "1.0.0"

from .client import MCPClient
from .cli import cli

__all__ = ["MCPClient", "cli", "__version__"]