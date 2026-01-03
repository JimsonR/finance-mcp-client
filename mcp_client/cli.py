import json
import click
from loguru import logger

from .client import MCPClient


@click.group()
def cli():
    """MCP Client CLI"""
    pass


# --------------------------------------------------
# TEST COMMAND
# --------------------------------------------------

@cli.command()
@click.option("--transport", default="stdio", help="Transport mode (stdio/http)")
@click.option("--host", default="127.0.0.1", help="Server host (HTTP mode)")
@click.option("--port", default=8080, help="Server port (HTTP mode)")
@click.option("--url", help="Full URL for deployed server (overrides host/port)")
def test(transport, host, port, url):
    """Test MCP client connection"""
    client = MCPClient(transport, host, port, url)

    try:
        if client.connect():
            logger.info("✓ Connection successful")

            # Test tools
            tools = client.list_tools()
            logger.info(f"✓ Found {len(tools)} tools: {[t['name'] for t in tools]}")

            # Try search_modules tool if available
            search_tool = next(
                (t for t in tools if t["name"] == "search_modules"), None
            )

            if search_tool:
                result = client.call_tool(
                    "search_modules",
                    {"query": "test", "limit": 1},
                )
                logger.info("✓ search_modules tool working")

    except Exception as e:
        logger.error(f"✗ Test failed: {e}")

    finally:
        client.disconnect()


# --------------------------------------------------
# SEARCH COMMAND
# --------------------------------------------------

@cli.command()
@click.option("--transport", default="stdio", help="Transport mode (stdio/http)")
@click.option("--host", default="127.0.0.1", help="Server host (HTTP mode)")
@click.option("--port", default=8080, help="Server port (HTTP mode)")
@click.option("--url", help="Full URL for deployed server (overrides host/port)")
@click.argument("query")
def search(transport, host, port, url, query):
    """Search using MCP server"""
    client = MCPClient(transport, host, port, url)

    try:
        if client.connect():
            result = client.call_tool(
                "search_modules",
                {"query": query, "limit": 5},
            )

            if isinstance(result, dict) and result.get("elicitation"):
                print("Elicitation required:")
                print(result["message"])
            else:
                print(result)

    except Exception as e:
        logger.error(f"Search failed: {e}")

    finally:
        client.disconnect()


# --------------------------------------------------
# INTERACTIVE TOOL MODE
# --------------------------------------------------

@cli.command()
@click.option("--transport", default="stdio", help="Transport mode (stdio/http)")
@click.option("--host", default="127.0.0.1", help="Server host (HTTP mode)")
@click.option("--port", default=8080, help="Server port (HTTP mode)")
@click.option("--url", help="Full URL for deployed server (overrides host/port)")
@click.argument("tool_name")
def interactive(transport, host, port, url, tool_name):
    """Run a tool in interactive elicitation mode"""
    client = MCPClient(transport, host, port, url)

    try:
        if client.connect():
            print(f"Starting interactive session with tool: {tool_name}")

            print("Enter initial arguments as JSON (or {} for empty):")
            user_input = input("> ")

            try:
                arguments = json.loads(user_input) if user_input.strip() else {}
            except json.JSONDecodeError:
                logger.error("Invalid JSON format")
                return

            result = client.call_tool_interactive(tool_name, arguments)
            print("\nFinal result:")
            print(result)

    except Exception as e:
        logger.error(f"Interactive session failed: {e}")

    finally:
        client.disconnect()


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    cli()
