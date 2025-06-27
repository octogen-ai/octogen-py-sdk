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
    SystemMessage,
)
from langchain_core.output_parsers import JsonOutputParser
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from octogen.shop_agent.base import ShopAgent
from showcase.schema import (
    AgentResponse,
    HydratedAgentResponse,
)

logger = structlog.get_logger()


def process_agent_response(
    unhydrated_response: AgentResponse, messages: Sequence[BaseMessage]
) -> str:
    """Process the agent response."""
    # No transformation needed, just return the JSON representation
    hydrated_response = HydratedAgentResponse(**unhydrated_response.model_dump())
    return hydrated_response.model_dump_json()


@asynccontextmanager
async def create_feed_agent(
    model: BaseChatModel,
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> AsyncGenerator[ShopAgent, None]:
    """Create a feed agent that doesn't require MCP connection."""
    # Use InMemorySaver if no checkpointer provided
    if not checkpointer:
        checkpointer = InMemorySaver()

    # Create a JSON output parser for format instructions
    response_parser = JsonOutputParser(pydantic_object=AgentResponse)
    format_instructions = response_parser.get_format_instructions()

    try:
        # Try loading from hub if available
        system_prompt = hub.pull("feed-agent")
        system_messages = system_prompt.invoke(
            {"format_instructions": format_instructions}
        ).messages
    except Exception as e:
        logger.warning(f"Could not load prompt from hub: {e}. Using fallback prompt.")
        # Fallback system message if hub is not available
        system_messages = [
            SystemMessage(
                content=f"""You are a helpful e-commerce shopping assistant whose primary goal is to guide users through their shopping discovery journey. You help users find exactly what they're looking for by understanding their needs and translating them into effective search queries.

## Your Role:
- Be attentive and helpful, focusing on the user's shopping needs
- Listen carefully to what the user is looking for and any preferences they mention
- Create a welcoming, conversational experience through your preamble responses
- Guide users to discovering products they'll love by crafting thoughtful searches

## Search Query Structure:
You'll create search queries following our CatalogTextSearchParams structure:
- text: The main search query text (Required)
- facets: List of facet filters with supported types:
  * brand_name: Filter by brands (e.g., "nike", "adidas")
  * product_type: Filter by product types (e.g., "shoes", "t-shirts")
- limit: Maximum number of results to return (optional)
- price_min/price_max: Price range filters (optional)

## Guidelines:
- Respond with a helpful, conversational preamble that acknowledges the user's request
- Create a search_query type response with parameters that will best serve the user's needs
- DO NOT make any tool calls
- DO NOT recommend products - only return search queries

{format_instructions}
"""
            )
        ]

    # Create agent directly without MCP tools
    agent = ShopAgent(
        model=model,
        tools=[],  # No tools
        system_message=system_messages,
        response_class=AgentResponse,
        hydrated_response_class=HydratedAgentResponse,
        rec_expansion_fn=process_agent_response,
        checkpointer=checkpointer,
    )

    try:
        yield agent
    finally:
        # Any cleanup if needed
        pass
