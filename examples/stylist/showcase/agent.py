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

from octogen.api.types.search_tool_output import Product
from octogen.shop_agent.base import ShopAgent
from octogen.shop_agent.factory import create_agent
from octogen.shop_agent.utils import expand_ai_recommendations
from showcase.schema import (
    HydratedOutfit,
    HydratedStylistAgentResponse,
    HydratedStylistRecommendation,
    StylistAgentResponse,
)

logger = structlog.get_logger()


def process_product_recommendations(
    unhydrated_response: StylistAgentResponse, messages: Sequence[BaseMessage]
) -> str:
    """Process the response to expand product recommendations."""
    if (
        unhydrated_response.product_recommendations is not None
        or unhydrated_response.current_outfit is not None
    ):
        hydrated_response = HydratedStylistAgentResponse(
            response_type=unhydrated_response.response_type,
            preamble=unhydrated_response.preamble,
        )
        hydrated_recommendations = []
        if unhydrated_response.product_recommendations is not None:
            unhydrated_recommendations = unhydrated_response.product_recommendations
            for recommendation in unhydrated_recommendations:
                expanded_products = expand_ai_recommendations(
                    list(messages),
                    "style_and_tags_search_products",
                    [product.model_dump() for product in recommendation.products],
                )
                hydrated_recommendations.append(
                    HydratedStylistRecommendation(
                        group_summary=recommendation.group_summary,
                        products=[Product(**product) for product in expanded_products],
                    )
                )
            hydrated_response.product_recommendations = hydrated_recommendations
        if unhydrated_response.current_outfit is not None:
            hydrated_outfit: HydratedOutfit = {}
            for product_type, product in unhydrated_response.current_outfit.items():
                if product is None:
                    hydrated_outfit[product_type] = None
                    continue
                expanded_product = expand_ai_recommendations(
                    list(messages),
                    "style_and_tags_search_products",
                    [product.model_dump()],
                )
                if len(expanded_product) > 0:
                    hydrated_outfit[product_type] = Product(**expanded_product[0])
                    logger.info(
                        "Successfully expanded product",
                        product_type=product_type,
                        product_keys=expanded_product[0].keys(),
                    )
                else:
                    # If we are unable to expand the product, use the original product and log a warning
                    logger.warning(
                        "Unable to expand product",
                        product_type=product_type,
                        product=product,
                    )
                    hydrated_outfit[product_type] = Product(**product.model_dump())
            if hydrated_outfit:
                logger.debug(
                    "Hydrated outfit",
                    hydrated_outfit_product_keys=[
                        p.model_dump().keys()
                        for p in hydrated_outfit.values()
                        if p is not None
                    ],
                )
                hydrated_response.current_outfit = hydrated_outfit
        return hydrated_response.model_dump_json()
    else:
        return unhydrated_response.model_dump_json()


@asynccontextmanager
async def create_stylist_agent(
    model: BaseChatModel,
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> AsyncGenerator[ShopAgent, None]:
    """Create a stylist agent."""
    async with create_agent(
        model=model,
        agent_name="Stylist",
        response_class=StylistAgentResponse,
        hydrated_response_class=HydratedStylistAgentResponse,
        rec_expansion_fn=process_product_recommendations,
        tool_names=["style_and_tags_search_products", "enrich_product_image"],
        hub_prompt_id="stylist_agent",
        checkpointer=checkpointer,
    ) as agent:
        yield agent
