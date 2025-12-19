#!/usr/bin/env python
import os
import logging
import argparse
import asyncio
from contextlib import AsyncExitStack
from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
#from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, SseConnectionParams
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
from rich.console import Console
from rich.logging import RichHandler
from rich.prompt import Prompt
from rich.theme import Theme
from google.adk.agents import Agent
from google.adk.tools import AgentTool

# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Configure rich logging
logging.basicConfig(
    level=logging.ERROR,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger(__name__)

# Define custom theme
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "yellow",
    "danger": "bold red",
    "success": "bold green",
    "query": "bold blue",
    "response": "bold green",
    "assistant": "bold magenta",
    "user": "bold magenta"
})

# Initialize rich console with custom theme
console = Console(theme=custom_theme)

load_dotenv()

# Define constants for Runner
APP_NAME = "zerodha_mcp_client"
USER_ID = "cli_user"
SESSION_ID = "cli_session"

#### Create a Search sub-agent ###############
search_agent = Agent(
    name="search_agent",
    model="gemini-3-flash-preview",
    description=(
        "Agent to answer general questions using google search"
    ),
    instruction=(
        "You are a helpful agent who can answer fundamental analysis for the input company or stock"
    ),
    tools=[google_search],
)
#### Wrap the sub-agent in AgentTool ###############
search_agent_tool = AgentTool(agent=search_agent, skip_summarization=False)

async def main():
    parser = argparse.ArgumentParser(description='MCP Client using Google ADK')
    parser.add_argument('--host', help='MCP server host (default: localhost)')
    parser.add_argument('--port', type=int, help='MCP server port (default: 8001)')
    args = parser.parse_args()

    mcp_host = args.host or os.environ.get("MCP_HOST", "localhost")
    mcp_port = args.port or int(os.environ.get("MCP_PORT", "8001"))
    mcp_url = f"http://{mcp_host}:{mcp_port}/sse"

    exit_stack = AsyncExitStack()

    try:
        # Use MCPToolset to connect and get tools
        logger.debug(f"[info]Connecting to MCP server at {mcp_url} via MCPToolset...[/info]")
        tools1 = McpToolset(
            connection_params=SseConnectionParams(url=mcp_url)
        )
        # Push the toolset's exit stack onto our main exit stack for cleanup
        # await exit_stack.enter_async_context(toolset_exit_stack)

        # No need to call get_tools() again, 'tools' is already the list
        # logger.debug(f"[info]MCPToolset loaded {len(tools)} tools from server[/info]")

        # Create the Google ADK agent
        agent = LlmAgent(
            name="zerodha_trading_assistant",
            model="gemini-3-flash-preview", # Ensure this model is available/correct
            description="Zerodha Trading Account Assistant via MCP",
            instruction="""
You are a Zerodha Trading Account Assistant, helping users securely manage their accounts, orders, portfolio, and positions using tools provided over MCP.

# Important Instructions:
- ALWAYS respond in plain text. NEVER use markdown formatting (no asterisks, hashes, or code blocks).
- Respond in human-like conversational, friendly, and professional tone in concise manner.

# Authentication Steps (must be followed if no access token is generated):
1. Use the 'get_login_url' tool to generate a Kite login URL and ask the user to log in and send the request token to you. Use this tool automatically when the user is not authenticated.
2. Use the 'get_access_token' tool with the request token to generate and validate the access token.
3. Proceed only if the access token is valid.

# Responsibilities:
- Check if the user is authenticated (e.g., by calling 'get_user_profile').
- Assist with order placement ('place_order'), modification ('modify_order'), and cancellation ('cancel_order').
- Provide insights on portfolio holdings ('get_holdings'), positions ('get_positions'), and available margin ('get_margins').
- Track order status ('get_orders'), execution details ('get_order_trades'), and trade history ('get_order_history').
- You can use search_agent_tool tool to search for information on the internet for any company or stock.
- Any more tools can be used if needed.

# Limitations:
# You do not provide real-time market quotes, historical data, or financial advice. Your role is to ensure secure, efficient, and compliant account management.
""",
            tools=[tools1,search_agent_tool],
        )

        # Initialize SessionService and Runner
        session_service = InMemorySessionService()
        runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

        # Create Session
        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )

        console.print(f"[info]Session created: {session}[/info]")

        # Welcome message
        console.print()
        console.print("[info]Welcome to Zerodha! (Running in non-live test mode)[/info]", style="response")

        # Main user input loop
        while True:
            console.print() # Spacing before prompt
            try:
                # No need for asyncio.to_thread for non-live prompt
                user_query = Prompt.ask("[query]Enter your query:[/query] [dim](or 'quit' to exit)[/dim]")
            except EOFError:
                user_query = 'quit'

            if user_query.lower() == 'quit':
                break

            console.print(f"[user]You:[/user] {user_query}")
            console.print()

            user_content = types.Content(role='user', parts=[types.Part.from_text(text=user_query)])

            final_response_text = ""
            try:
                events_async = runner.run_async(
                    session_id=session.id,
                    user_id=USER_ID,
                    new_message=user_content
                )
                async for event in events_async:
                     if event.is_final_response() and event.content:
                         for part in event.content.parts:
                            if part.text:
                                final_response_text += part.text

            except Exception as runner_ex:
                 logger.error(f"[danger]Error during runner execution: {runner_ex}[/danger]", exc_info=True)
                 final_response_text = f"[bold red]Error: {runner_ex}[/bold red]"

            # Print the final accumulated response
            console.print(f"[assistant]Assistant:[/assistant] {final_response_text}", style="response")
            console.print() # Add extra spacing

    except Exception as e:
        logger.error(f"[danger]Error in main execution loop: {e}[/danger]", exc_info=True)
    finally:
        await exit_stack.aclose()
        logger.debug("[info]MCPToolset connection closed and resources cleaned up.[/info]")
        console.print("[info]Exiting client.[/info]")

if __name__ == "__main__":
    asyncio.run(main())