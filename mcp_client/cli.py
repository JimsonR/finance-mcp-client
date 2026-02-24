import json
import os
import click
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv
from loguru import logger

from .client import MCPClient
from .streamable_http_client import MCPStreamableHTTPClient


@click.group()
def cli():
    """MCP Client CLI"""
    load_dotenv()


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
# LOCAL HTTPBIN TOOLS
# --------------------------------------------------


@cli.command(name="httpbin-ip")
def httpbin_ip():
    """Call https://httpbin.org/ip and print caller IP"""
    response = requests.get("https://httpbin.org/ip", timeout=30)
    response.raise_for_status()
    data = response.json()
    print(json.dumps(data, indent=2))


@cli.command(name="httpbin-echo")
@click.option("--message", required=True, help="Message to echo back")
@click.option("--meta", default="{}", help="Optional JSON metadata")
def httpbin_echo(message, meta):
    """POST to https://httpbin.org/post and echo payload"""
    try:
        meta_obj = json.loads(meta) if meta.strip() else {}
    except json.JSONDecodeError:
        raise click.ClickException("--meta must be valid JSON")

    payload = {"message": message, "meta": meta_obj}

    response = requests.post(
        "https://httpbin.org/post",
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    print(json.dumps(data, indent=2))


# --------------------------------------------------
# MISTRAL CHAT HELPER
# --------------------------------------------------


@cli.command(name="mistral-chat")
@click.option("--prompt", required=True, help="User prompt to send to Mistral")
@click.option("--model", default="mistral-small-latest", show_default=True)
@click.option("--max-tokens", default=2048, show_default=True, help="Max tokens for response")
@click.option("--temperature", default=0.2, show_default=True)
@click.option(
    "--mcp-url",
    default=lambda: os.getenv("MCP_URL", "http://localhost:7071/api/mcp"),
    show_default=True,
    help="HTTP MCP server endpoint for tools/list + tools/call",
)
@click.option(
    "--tool-choice",
    default="auto",
    show_default=True,
    help="auto|none|<tool_name> (force a specific tool)",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Print extra debug info (tools + raw model responses)",
)
def mistral_chat(prompt, model, max_tokens, temperature, mcp_url, tool_choice, debug):
    """Call Mistral chat completions and route tool calls to an MCP HTTP server."""

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise click.ClickException(
            "MISTRAL_API_KEY is not set (use your .env or shell env)."
        )

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # ---- Connect to MCP server using StreamableHTTP client ----
    # Parse the MCP URL into base_url + mcp_path for MCPStreamableHTTPClient
    parsed = urlparse(mcp_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    mcp_path = parsed.path or "/mcp"

    mcp_client = MCPStreamableHTTPClient(
        base_url=base_url,
        mcp_path=mcp_path,
        timeout=120,
    )
    if not mcp_client.connect():
        raise click.ClickException(f"Failed to connect to MCP server at {mcp_url}.")

    mcp_tools = mcp_client.list_tools()

    def _schema_from_tool(t):
        schema = t.get("inputSchema") or t.get("input_schema")
        if isinstance(schema, dict):
            return schema
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        }

    tools = []
    for t in mcp_tools:
        name = t.get("name")
        if not name:
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": t.get("description") or "",
                    "parameters": _schema_from_tool(t),
                },
            }
        )

    def call_tool(name, arguments):
        result = mcp_client.call_tool(name, arguments or {})
        # MCPStreamableHTTPClient may return dict (with visualization) or str
        if isinstance(result, dict):
            # If it has content, extract it; otherwise JSON-encode the whole thing
            content = result.get("content")
            if content is not None:
                return content if isinstance(content, str) else json.dumps(content)
            return json.dumps(result)
        return result if isinstance(result, str) else json.dumps(result)

    if tool_choice not in ("auto", "none"):
        resolved_tool_choice = {
            "type": "function",
            "function": {"name": tool_choice},
        }
    else:
        resolved_tool_choice = tool_choice

    if debug:
        logger.info(f"MCP URL: {mcp_url}")
        logger.info(f"Advertised MCP tools: {[t.get('name') for t in mcp_tools if t.get('name')]} ")
        logger.info(f"Mistral tool_choice: {resolved_tool_choice}")

    messages = [
        {
            "role": "system",
            "content": (
                "You have access to tools exposed by an MCP server. "
                "If the user asks for live/external information (like IP address), call an appropriate tool "
                "instead of explaining how to do it.\n\n"
                "IMPORTANT: For data/SQL questions you MUST follow this two-step workflow:\n"
                "1. First call get_sql_schema to discover the relevant tables and columns.\n"
                "2. Then call run_sql_query with a SQL query built from that schema to get the actual data.\n"
                "Never stop after just getting the schema — always follow up with run_sql_query to answer the user's question.\n"
                "If the schema result is missing tables you need (e.g. customers, orders), call get_sql_schema again "
                "with different search_terms that include the missing table names."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    for round_num in range(10):  # allow up to 10 tool-call rounds
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": resolved_tool_choice,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if debug:
            logger.debug(f"--- Round {round_num + 1} ---")
            logger.debug(f"Messages count: {len(messages)}")

        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()

        if debug:
            logger.debug("Raw Mistral response (truncated): " + json.dumps(data)[:2000])

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")

        if finish_reason == "length":
            logger.warning(
                "Response truncated due to max_tokens limit. "
                "Increase --max-tokens or simplify the prompt."
            )

        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            logger.info(f"Mistral requested {len(tool_calls)} tool call(s)")

            # Append the FULL assistant message once (with content + all tool_calls)
            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": tool_calls,
                }
            )

            # Execute each tool and append results
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name")
                raw_args = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError as e:
                    raise click.ClickException(
                        f"Tool arguments for {name} were not valid JSON: {e}. "
                        f"Raw args: {raw_args[:200]}. "
                        "This often means max_tokens was too low and the response was truncated."
                    )

                result = call_tool(name, args or {})

                if debug:
                    logger.debug(f"Tool {name} args: {args}")
                    logger.debug(
                        "Tool result (truncated): "
                        + (result[:500] if isinstance(result, str) else json.dumps(result)[:500])
                    )

                messages.append(
                    {
                        "role": "tool",
                        "name": name,
                        "tool_call_id": tc.get("id"),
                        "content": result if isinstance(result, str) else json.dumps(result),
                    }
                )
            # loop again to let the model use the tool results
            continue

        content = message.get("content")
        if content:
            print(content)
            return

        # Fallback: dump raw response
        print(json.dumps(data, indent=2))
        return

    raise click.ClickException("Max tool-call rounds exceeded without a final answer.")


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    cli()
