from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Optional, Sequence, Type

import structlog
from langchain import hub
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_mcp_adapters.tools import load_mcp_tools  # type: ignore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel

from octogen.shop_agent.base import ShopAgent
from octogen.shop_agent.settings import AgentSettings, get_agent_settings

logger = structlog.get_logger()


@asynccontextmanager
async def create_agent(
    model: BaseChatModel,
    agent_name: str,
    response_class: Type[BaseModel],
    hydrated_response_class: Type[BaseModel],
    rec_expansion_fn: Callable[[Any, Sequence[BaseMessage]], str],
    tool_names: list[str],
    hub_prompt_id: str,
    additional_prompt_args: Optional[dict[str, Any]] = None,
    agent_settings: AgentSettings = get_agent_settings(),
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> AsyncGenerator[ShopAgent, None]:
    """Generic factory function for creating a shop agent.

    Args:
        model: The LLM model to use for the agent
        agent_name: Human-readable name of the agent (for logging)
        response_class: Pydantic model for agent responses
        hydrated_response_class: Pydantic model for expanded responses
        rec_expansion_fn: Function to expand product recommendations
        tool_names: List of MCP tool names to use
        hub_prompt_id: LangChain Hub ID for the system prompt
        additional_prompt_args: Additional args to pass to the prompt template
        mcp_settings: MCP settings
        checkpointer: Optional checkpoint saver for the agent

    Yields:
        Configured ShopAgent instance
    """
    if not checkpointer:
        checkpointer = InMemorySaver()

    logger.info(f"Creating {agent_name} agent")
    async with streamablehttp_client(
        url=agent_settings.mcp_server_url,
        timeout=agent_settings.mcp_server_timeout,
        headers=agent_settings.mcp_auth_header,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Create response parser for format instructions
            response_parser = JsonOutputParser(pydantic_object=response_class)

            # Build prompt args
            prompt_args = {
                "format_instructions": response_parser.get_format_instructions()
            }
            if additional_prompt_args:
                prompt_args.update(additional_prompt_args)

            # Load system prompt
            system_prompt = hub.pull(hub_prompt_id).invoke(prompt_args).messages

            # Get tools
            tools = await load_mcp_tools(session)

            # Filter tools by name
            filtered_tools = [tool for tool in tools if tool.name in tool_names]

            if not filtered_tools:
                logger.warning(f"No tools found matching {tool_names}")

            # Create agent
            agent = ShopAgent(
                model=model,
                tools=filtered_tools,
                system_message=system_prompt,
                response_class=response_class,
                hydrated_response_class=hydrated_response_class,
                rec_expansion_fn=rec_expansion_fn,
                checkpointer=checkpointer,
            )

            yield agent
