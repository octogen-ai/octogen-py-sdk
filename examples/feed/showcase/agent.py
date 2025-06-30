from contextlib import asynccontextmanager
from typing import (
    AsyncGenerator,
    Optional,
    Sequence,
)

import structlog
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
)
from langgraph.checkpoint.base import BaseCheckpointSaver

from octogen.shop_agent.base import ShopAgent
from octogen.shop_agent.factory import create_agent
from showcase.schema import (
    AgentResponse,
)

logger = structlog.get_logger()


def process_agent_response(
    unhydrated_response: AgentResponse, messages: Sequence[BaseMessage]
) -> str:
    """Process the agent response."""
    # No transformation needed, just return the JSON representation
    return unhydrated_response.model_dump_json()


@asynccontextmanager
async def create_feed_agent(
    model: BaseChatModel,
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> AsyncGenerator[ShopAgent, None]:
    """Create a stylist agent."""
    async with create_agent(
        model=model,
        agent_name="Feed Agent",
        response_class=AgentResponse,
        hydrated_response_class=AgentResponse,
        rec_expansion_fn=process_agent_response,
        tool_names=["enrich_product_image"],
        hub_prompt_id="feed-agent",
        checkpointer=checkpointer,
    ) as agent:
        yield agent
