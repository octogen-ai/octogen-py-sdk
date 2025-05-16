import json
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
from octogen.shop_agent.utils import (
    expand_ai_recommendations,
)
from showcase.schema import (
    ComparisonResponse,
    HydratedComparisonDataCategory,
    HydratedComparisonResponse,
)

logger = structlog.get_logger()


def process_product_recommendations(
    unhydrated_response: ComparisonResponse, messages: Sequence[BaseMessage]
) -> str:
    """Process the response to expand product recommendations."""
    if (
        unhydrated_response.response_type == "comparison"
        and unhydrated_response.comparison_data is not None
    ):
        hydrated_response = HydratedComparisonResponse(
            **unhydrated_response.model_dump()
        )
        unhydrated_recommendations = unhydrated_response.comparison_data
        hydrated_recommendations = []
        for recommendation in unhydrated_recommendations:
            expanded_products = expand_ai_recommendations(
                list(messages),
                "agent_search_products",
                [product.model_dump() for product in recommendation.items],
            )
            hydrated_recommendations.append(
                HydratedComparisonDataCategory(
                    category=recommendation.category,
                    items=[Product(**product) for product in expanded_products],
                )
            )
        hydrated_response.comparison_data = hydrated_recommendations
        return hydrated_response.model_dump_json()
    else:
        return unhydrated_response.model_dump_json()


def get_reduced_product_schema() -> str:
    """Extract important fields from the Product schema."""
    # Get the full schema
    full_schema = Product.model_json_schema()

    # Define the important fields we want to keep
    important_fields = [
        "uuid",
        "name",
        "description",
        "brand_name",
        "current_price",
        "original_price",
        "url",
        "image",
        "images",
        "aggregateRating",
        "catalog",
        "categories",
        "materials",
        "sizes",
        "color_info",
        "dimensions",
        "fit",
        "patterns",
        "audience",
        "tags",
        "rating",
    ]

    # Create a filtered schema with only the fields we care about
    filtered_schema = {
        "title": full_schema.get("title", "SearchProduct"),
        "type": "object",
        "description": "Product information with key fields for comparison",
        "properties": {},
        "required": full_schema.get("required", ["uuid"]),
    }

    # Extract just the properties we want
    all_properties = full_schema.get("properties", {})
    for field in important_fields:
        if field in all_properties:
            # Add the original property definition
            filtered_schema["properties"][field] = all_properties[field]

            # Add a description if missing
            if "description" not in filtered_schema["properties"][field]:
                filtered_schema["properties"][field]["description"] = (
                    f"{field} of the product"
                )

    return json.dumps(filtered_schema, indent=2)


@asynccontextmanager
async def create_comparison_agent(
    model: BaseChatModel,
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> AsyncGenerator[ShopAgent, None]:
    """Load the comparison agent with tools."""
    additional_prompt_args = {
        "product_schema": get_reduced_product_schema(),
    }

    async with create_agent(
        model=model,
        agent_name="Comparison",
        response_class=ComparisonResponse,
        hydrated_response_class=HydratedComparisonResponse,
        rec_expansion_fn=process_product_recommendations,
        tool_names=["agent_search_products"],
        hub_prompt_id="example_comparison_agent",
        additional_prompt_args=additional_prompt_args,
        checkpointer=checkpointer,
    ) as agent:
        yield agent
