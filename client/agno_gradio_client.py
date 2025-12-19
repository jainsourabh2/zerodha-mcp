#!/usr/bin/env python
import os
import sys
import asyncio
import logging
import gradio as gr
from agno.models.openai import OpenAIChat
from agno.agent import Agent
from agno.tools.mcp import MCPTools
from mcp import ClientSession
from mcp.client.sse import sse_client
from dotenv import load_dotenv
from typing import Optional, Dict, List

# Silence all logging
class SilentFilter(logging.Filter):
    def filter(self, record):
        return False

# Configure root logger to be silent
root_logger = logging.getLogger()
root_logger.addFilter(SilentFilter())
root_logger.setLevel(logging.DEBUG)

# Silence specific loggers
for logger_name in ['agno', 'httpx', 'urllib3', 'asyncio']:
    logger = logging.getLogger(logger_name)
    logger.addFilter(SilentFilter())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

load_dotenv()

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self._streams_context = None
        self._session_context = None

    async def connect_to_sse_server(self, server_url: str):
        """Connect to an MCP server running with SSE transport"""
        try:
            self._streams_context = sse_client(url=server_url)
            streams = await self._streams_context.__aenter__()

            self._session_context = ClientSession(*streams)
            self.session: ClientSession = await self._session_context.__aenter__()

            await self.session.initialize()
            await self.session.list_tools()
            return True
        except Exception as e:
            await self.cleanup()
            raise e

    async def cleanup(self):
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)
        self.session = None

    async def disconnect(self):
        await self.cleanup()

class ZerodhaAssistant:
    def __init__(self):
        self.client: Optional[MCPClient] = None
        self.agent: Optional[Agent] = None
        self.connected = False

    async def connect(self, host: str, port: int) -> str:
        """Connect to MCP server and initialize agent"""
        if self.connected:
            return "Already connected!"

        try:
            mcp_url = f"http://{host}:{port}/sse"
            self.client = MCPClient()
            await self.client.connect_to_sse_server(mcp_url)

            # Initialize tools and agent
            mcp_tools = MCPTools(session=self.client.session)
            await mcp_tools.initialize()

            self.agent = Agent(
                instructions="""
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
- Any more tools can be used if needed.

# Limitations:
You do not provide real-time market quotes, historical data, or financial advice. Your role is to ensure secure, efficient, and compliant account management.
""",
                model=OpenAIChat(id="gpt-4o"),
                tools=[mcp_tools],
                show_tool_calls=False,
                markdown=False,
                add_history_to_messages=True,
                num_history_responses=10,
                read_tool_call_history=True,
                read_chat_history=True,
                tool_call_limit=10,
                telemetry=False,
                add_datetime_to_instructions=True
            )

            self.connected = True
            return "Connected successfully! Ready to assist you."
        except Exception as e:
            if self.client:
                await self.client.disconnect()
            self.client = None
            self.agent = None
            self.connected = False
            return f"Failed to connect: {str(e)}"

    async def disconnect(self) -> str:
        """Disconnect from MCP server"""
        if not self.connected:
            return "Not connected!"

        try:
            if self.client:
                await self.client.disconnect()
            self.client = None
            self.agent = None
            self.connected = False
            return "Disconnected successfully!"
        except Exception as e:
            return f"Error during disconnect: {str(e)}"

    async def chat(self, message: str, history: List[List[str]]) -> str:
        """Process a chat message"""
        if not self.connected or not self.agent:
            return "Not connected to MCP server! Please connect first."

        try:
            response = await self.agent.arun(message, stream=False)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"

# Create assistant instance
assistant = ZerodhaAssistant()

async def connect_handler(host: str, port: str) -> str:
    """Handle connection button click"""
    try:
        port_num = int(port)
        return await assistant.connect(host, port_num)
    except ValueError:
        return "Invalid port number!"
    except Exception as e:
        return f"Connection error: {str(e)}"

async def disconnect_handler() -> str:
    """Handle disconnection button click"""
    return await assistant.disconnect()

async def chat_handler(message: str, history: List[List[str]]) -> List[List[str]]:
    """Handle chat messages"""
    if not assistant.connected or not assistant.agent:
        return history + [[message, "Not connected to MCP server! Please connect first."]]

    try:
        response = await assistant.agent.arun(message, stream=False)
        return history + [[message, response.content]]
    except Exception as e:
        return history + [[message, f"Error: {str(e)}"]]

# Create the Gradio interface
with gr.Blocks(title="Zerodha Trading Assistant", theme=gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter")
)) as demo:
    gr.Markdown(
        """
        <div style="text-align: center; margin-bottom: 1rem">
            <h1 style="margin-bottom: 0.5rem">🤖 Zerodha AI Trading Assistant</h1>
            <p style="margin: 0; opacity: 0.8">Manage your Zerodha account via secure MCP connection</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 🔌 MCP Connection")
            with gr.Group():
                status_box = gr.Textbox(
                    label="Status",
                    placeholder="Not connected",
                    interactive=False,
                    show_label=False,
                )
                with gr.Row():
                    host_input = gr.Textbox(
                        label="Host",
                        value=os.environ.get("MCP_HOST", "localhost"),
                        lines=1,
                        scale=7,
                        container=False,
                    )
                    port_input = gr.Textbox(
                        label="Port",
                        value=os.environ.get("MCP_PORT", "8001"),
                        lines=1,
                        scale=3,
                        container=False,
                    )
                with gr.Row():
                    connect_btn = gr.Button("Connect", variant="primary", scale=2)
                    disconnect_btn = gr.Button("Disconnect", scale=1)

    with gr.Row():
        chatbot = gr.Chatbot(
            value=[],
            label="Chat",
            height=450,
            show_label=False,
            avatar_images=["https://api.dicebear.com/7.x/bottts/svg?seed=user", "https://api.dicebear.com/7.x/bottts/svg?seed=assistant"],
            bubble_full_width=False,
        )

    with gr.Row():
        msg_box = gr.Textbox(
            label="Message",
            placeholder="Connect to MCP server first...",
            lines=1,
            max_lines=1,
            show_label=False,
            scale=20,
            container=False,
            interactive=False,  # Start as disabled
        )
        send_btn = gr.Button("Send", scale=3, size="sm", interactive=False)  # Start as disabled

    # Set up event handlers
    def enable_chat(status_text):
        """Enable/disable chat based on connection status"""
        is_connected = "Connected successfully" in status_text
        return {
            msg_box: gr.update(
                interactive=is_connected,
                placeholder="Type your message here..." if is_connected else "Connect to MCP server first..."
            ),
            send_btn: gr.update(interactive=is_connected),
        }

    connect_btn.click(
        fn=connect_handler,
        inputs=[host_input, port_input],
        outputs=status_box,
    ).then(
        fn=enable_chat,
        inputs=[status_box],
        outputs=[msg_box, send_btn],
    )

    disconnect_btn.click(
        fn=disconnect_handler,
        inputs=[],
        outputs=status_box,
    ).then(
        fn=enable_chat,
        inputs=[status_box],
        outputs=[msg_box, send_btn],
    )

    # Chat submission can happen through either the textbox or send button
    msg_box.submit(
        fn=chat_handler,
        inputs=[msg_box, chatbot],
        outputs=[chatbot],
        show_progress=True,
    ).then(
        fn=lambda: "",  # Clear input after sending
        inputs=None,
        outputs=msg_box,
    )

    send_btn.click(
        fn=chat_handler,
        inputs=[msg_box, chatbot],
        outputs=[chatbot],
        show_progress=True,
    ).then(
        fn=lambda: "",  # Clear input after sending
        inputs=None,
        outputs=msg_box,
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Listen on all interfaces
        show_api=False,
        share=False,  # Set to True if you want to generate a public URL
    )