import json
from typing import Annotated, Any, Dict, List, Sequence, Type, TypedDict, Union

import structlog
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
)
from langchain_core.messages.ai import AIMessage
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep
from pydantic import BaseModel, Field, ValidationError

from octogen.api.types.search_tool_output import Product, SearchToolOutput


class ProductRecommendation(Product):
    justification: str


class ShopAgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_last_step: IsLastStep


class ShopAgentConfig(BaseModel):
    """Configuration for the shop agent."""

    user_id: str = Field(default="")
    thread_id: str = Field(default="")
    run_id: str = Field(default="")
    title: str = Field(
        default="", description="Title summarizing the conversation thread."
    )


logger = structlog.get_logger(__name__)


def get_tool_output(
    messages: List[BaseMessage], tool_name: str
) -> Union[str, dict, list[Union[str, dict]], None]:
    """
    Extracts tool output from a list of messages.

    Args:
        messages: List of messages to search through
        tool_name: Name of the tool to look for

    Returns:
        The content of the tool message, or None if not found
    """
    # Log the types of messages we're searching through
    # message_types = [type(m).__name__ for m in messages]
    # logger.debug(
    #     f"Searching for tool '{tool_name}' in {len(messages)} messages. Types: {message_types}"
    # )

    # Find all tool messages with matching name
    matching_tools = [
        m for m in messages if isinstance(m, ToolMessage) and m.name == tool_name
    ]
    all_products = SearchToolOutput(
        products=[],
        total_found=0,
        error=None,
    )

    if matching_tools:
        for potential_tool_message in reversed(messages):
            if not isinstance(potential_tool_message, ToolMessage):
                continue
            tool_message = potential_tool_message

            # If the content is a string and looks like a Python dict representation,
            # convert it to a proper dict
            if isinstance(tool_message.content, str):
                try:
                    result = SearchToolOutput.model_validate_json(tool_message.content)
                    all_products.products.extend(result.products)
                    all_products.total_found += result.total_found
                    all_products.error = result.error
                except Exception as e:
                    logger.warning(f"Failed to parse string as SearchToolOutput: {e}")
            elif (
                isinstance(tool_message.content, str)
                and tool_message.content.startswith("{")
                and not tool_message.content.startswith(
                    '{"'
                )  # Only use literal_eval if it's not JSON
                and "'" in tool_message.content
            ):
                try:
                    import ast

                    # Safely evaluate the Python literal expression
                    result = ast.literal_eval(tool_message.content)
                    # Check that the result is of an expected type
                    if isinstance(result, dict):
                        all_products.products.extend(result.get("products", []))
                        all_products.total_found += result.get("total_found", 0)
                        all_products.error = result.get("error", None)
                    else:
                        logger.warning(
                            f"Unexpected type from literal_eval: {type(result)}"
                        )
                except (SyntaxError, ValueError) as e:
                    logger.warning(
                        f"Failed to parse tool message as Python literal: {e}"
                    )
                    # Continue with the original content if parsing fails

    if all_products.total_found == 0:
        # Log all tool message names for debugging
        tool_messages = [m.name for m in messages if isinstance(m, ToolMessage)]
        if tool_messages:
            logger.debug(
                f"No ToolMessage found for {tool_name}. Available tools: {tool_messages}"
            )
        else:
            logger.debug(
                f"No ToolMessage found for {tool_name} and no tool messages in state"
            )

    logger.debug(
        f"Found {len(all_products.products)} products from {len(matching_tools)} tool messages"
    )

    return all_products.model_dump_json()


def get_product_recommendations(
    compact_list_of_products: List[Dict],
    aggregate_tool_output: Union[str, dict, list[Union[str, dict]]],
) -> List[dict]:
    """
    Process product recommendations by matching them with full product data.

    Args:
        list_of_products: List of product recommendations
        tool_output: Output from the product search tool

    Returns:
        List of expanded product recommendations
    """
    try:
        # Validate tool output
        if not aggregate_tool_output:
            logger.warning("No tool output found for get_products_for_client")
            return []

        # Extract products from tool output
        products_list = []
        if isinstance(aggregate_tool_output, str):
            try:
                # First check if it looks like a Python literal string (has single quotes)
                if (
                    aggregate_tool_output.startswith("{")
                    and not aggregate_tool_output.startswith('{"')
                    and "'" in aggregate_tool_output
                ):
                    import ast

                    # Safely evaluate the Python literal expression
                    parsed_output = ast.literal_eval(aggregate_tool_output)
                else:
                    # Try standard JSON parsing
                    parsed_output = json.loads(aggregate_tool_output)

                products_list = parsed_output.get("products", [])
            except (json.JSONDecodeError, SyntaxError, ValueError) as e:
                logger.warning(f"Failed to parse tool output format: {e}")
                # logger.debug(f"Problem tool output: {aggregate_tool_output[:100]}...")
                return []
        elif isinstance(aggregate_tool_output, dict):
            products_list = aggregate_tool_output.get("products", [])
        else:
            logger.warning(
                f"Unexpected tool output type: {type(aggregate_tool_output)}"
            )
            return []

        # Create lookup dictionary
        product_dict = {}
        for p in products_list:
            if isinstance(p, dict) and "uuid" in p:
                product_dict[p["uuid"]] = p

        # Process recommendations
        recs = []
        for p in compact_list_of_products:
            if not isinstance(p, dict):
                logger.debug(f"Skipping non-dict product: {p}")
                continue

            # Get product ID (either uuid or id)
            uuid = p.get("uuid") or p.get("id")
            if not uuid:
                logger.debug(f"Product missing uuid/id: {p}")
                continue

            # Find product in lookup dictionary
            if uuid in product_dict:
                try:
                    prod_dict = product_dict[uuid].copy()
                    # Add justification from the recommendation, if the llm provided one
                    prod_dict["justification"] = p.get("justification", "")
                    # Convert to ProductRecommendation and back to ensure valid schema
                    recs.append(
                        json.loads(ProductRecommendation(**prod_dict).model_dump_json())
                    )
                except Exception as e:
                    logger.warning(f"Error processing product {uuid}: {e}")
            else:
                logger.debug(f"Product with uuid {uuid} not found in tool outputs.")

        # Log warning if no recommendations were found
        if not recs:
            logger.debug("No recommendations found")

        return recs

    except Exception as e:
        logger.error(f"Error in get_product_recommendations: {e}")
        return []


def expand_ai_recommendations(
    messages: list[BaseMessage],
    tool_name: str,
    ai_recommendations: list[Dict[str, Any]],
) -> list[dict]:
    """
    Expand the AI recommendations into a list of dictionaries.

    Args:
        messages: list[BaseMessage] - The messages from conversation history
        tool_name: str - The name of the tool that was used to generate the recommendations
        ai_recommendations: list[Dict[str, Any]] - The recommendations from the AI after reading the tool outputs
    Returns:
        list[dict] - The expanded recommendations
    """
    aggregate_tool_output = get_tool_output(messages, tool_name)
    if aggregate_tool_output is None:
        logger.warning(
            f"No tool output found for {tool_name}. Returning original recommendations."
        )
        return ai_recommendations
    else:
        # Expand products
        expanded_products = get_product_recommendations(
            ai_recommendations, aggregate_tool_output
        )
        return expanded_products


def shrink_previous_recommendations(
    state: ShopAgentState,
    AgentResponseClass: Type[BaseModel],
) -> None:
    """
    Shrink all AI messages that are of recommendation response type to only contain uuid instead of SearchProduct.
    """
    for message in state["messages"]:
        if isinstance(message, AIMessage):
            if isinstance(message.content, str):
                body = (
                    message.content.replace("```json", "")
                    .replace("```", "")
                    .replace("\n", "")
                )
                try:
                    unhydrated_agent_response = AgentResponseClass.model_validate_json(
                        body
                    )
                    message.content = unhydrated_agent_response.model_dump_json()
                except ValidationError:
                    # logger.debug("Unable to validate AgentResponseClass for shrinking")
                    continue
