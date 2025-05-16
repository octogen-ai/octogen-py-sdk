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
from octogen.shop_agent.utils import expand_ai_recommendations
from showcase.schema import (
    AgentResponse,
    HydratedAgentResponse,
    ProductRecommendation,
)

logger = structlog.get_logger()


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
    async with create_agent(
        model=model,
        agent_name="Discovery",
        response_class=AgentResponse,
        hydrated_response_class=HydratedAgentResponse,
        rec_expansion_fn=process_product_recommendations,
        tool_names=["agent_search_products"],
        hub_prompt_id="discovery_agent",
        checkpointer=checkpointer,
    ) as agent:
        yield agent
