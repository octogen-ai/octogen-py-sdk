from contextlib import asynccontextmanager
from typing import (
    AsyncGenerator,
    Optional,
    Sequence,
)

import structlog
from langchain import hub
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_mcp_adapters.tools import load_mcp_tools  # type: ignore[import-untyped]
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from mcp import ClientSession
from mcp.client.sse import sse_client

from octogen.shop_agent.base import ShopAgent
from octogen.shop_agent.utils import expand_ai_recommendations
from showcase.schema import (
    AgentResponse,
    HydratedAgentResponse,
    ProductRecommendation,
)

logger = structlog.get_logger()

agent_response_parser = JsonOutputParser(pydantic_object=AgentResponse)


def process_product_recommendations(
    unhydrated_response: AgentResponse, messages: Sequence[BaseMessage]
) -> str:
    """Process the response to expand product recommendations."""
    if unhydrated_response.product_recommendations is not None:
        hydrated_response = HydratedAgentResponse(**unhydrated_response.model_dump())
        hydrated_recommendations = expand_ai_recommendations(
            list(messages),
            "agent_search_products",
            [
                product.model_dump()
                for product in unhydrated_response.product_recommendations
            ],
        )
        hydrated_response.product_recommendations = [
            ProductRecommendation(**product) for product in hydrated_recommendations
        ]
        return hydrated_response.model_dump_json()
    else:
        return unhydrated_response.model_dump_json()


@asynccontextmanager
async def create_discovery_agent(
    model: BaseChatModel,
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> AsyncGenerator[ShopAgent, None]:
    if not checkpointer:
        checkpointer = InMemorySaver()

    async with sse_client(url="http://0.0.0.0:8000/sse", timeout=60) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            system_prompt = (
                hub.pull("discovery_agent")
                .invoke(
                    dict(
                        format_instructions=agent_response_parser.get_format_instructions()
                    )
                )
                .messages
            )
            # Get tools
            tools = await load_mcp_tools(session)

            # Filter for style_and_tags_search
            style_tools = [
                tool for tool in tools if tool.name == "agent_search_products"
            ]
            agent = ShopAgent(
                model=model,
                tools=style_tools,
                system_message=system_prompt,
                response_class=AgentResponse,
                hydrated_response_class=HydratedAgentResponse,
                rec_expansion_fn=process_product_recommendations,
                checkpointer=checkpointer,
            )

            yield agent

            await session.close()
