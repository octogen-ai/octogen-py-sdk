import json
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

from octogen.api.types.search_tool_output import Product
from octogen.shop_agent.base import ShopAgent
from octogen.shop_agent.utils import (
    expand_ai_recommendations,
)
from showcase.schema import (
    ComparisonResponse,
    HydratedComparisonDataCategory,
    HydratedComparisonResponse,
)

logger = structlog.get_logger()

# Use a pretrained model

comparison_agent_response_parser = JsonOutputParser(pydantic_object=ComparisonResponse)


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
    """Load the stylist agent with tools."""
    if not checkpointer:
        checkpointer = InMemorySaver()

    async with sse_client(url="http://0.0.0.0:8000/sse", timeout=60) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            system_prompt = (
                hub.pull("comparison_agent")
                .invoke(
                    dict(
                        response_schema=comparison_agent_response_parser.get_format_instructions(),
                        product_schema=get_reduced_product_schema(),
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
                response_class=ComparisonResponse,
                hydrated_response_class=HydratedComparisonResponse,
                rec_expansion_fn=process_product_recommendations,
                checkpointer=checkpointer,
            )

            yield agent

            await session.close()
