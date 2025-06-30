import inspect
import json
import re
import time
import traceback
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
)

import jsonpatch  # type: ignore[import-untyped]
import structlog
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages.ai import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from pydantic import BaseModel

from octogen.shop_agent.utils import (
    ShopAgentConfig,
    ShopAgentState,
    shrink_previous_recommendations,
)

logger = structlog.get_logger()


def clean_json_comments(json_str: str) -> str:
    """Remove comments from JSON string.

    Args:
        json_str: The JSON string that may contain comments

    Returns:
        Clean JSON string with comments removed
    """
    # Remove single line comments (// comment)
    json_str = re.sub(r"//.*?$", "", json_str, flags=re.MULTILINE)
    # Remove multi-line comments (/* comment */)
    json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)
    return json_str


class ShopAgentModelNode:
    """Node to handle LLM responses and process product recommendations."""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        system_message: BaseMessage | List[BaseMessage],
        response_class: Type[BaseModel],
        rec_expansion_fn: Callable,
    ):
        self.llm_with_tools = llm.bind_tools(tools)
        self.logger = structlog.get_logger()
        if isinstance(system_message, BaseMessage):
            self.system_messages = [system_message]
        else:
            self.system_messages = system_message
        self.response_class = response_class
        self.rec_expansion_fn = rec_expansion_fn

    async def __call__(
        self, state: ShopAgentState, config: RunnableConfig
    ) -> Dict[str, Any]:
        shrink_previous_recommendations(state, self.response_class)
        messages = list(state["messages"])
        prompt_messages = self.system_messages + messages

        response = await self.llm_with_tools.ainvoke(prompt_messages, config)

        # Check if tool calls are present
        if hasattr(response, "tool_calls") and response.tool_calls:
            # This is an intermediate message with tool calls - just pass it through
            return {"messages": [response]}

            # Add the raw response to the state so it's available for processing

        updated_state = dict(state)
        updated_state["messages"] = list(state["messages"]) + [response]
        content = response.content
        if not isinstance(content, str):
            logger.warning("Response is not a string", response=response)
            return {"messages": [response]}
        try:
            # For markdown code blocks, extract the JSON
            if "```json" in content:
                self.logger.info("Found JSON in markdown code block")
                content = content.split("```json")[1].split("```")[0].strip()

            # Clean JSON comments before validation
            cleaned_content = clean_json_comments(content)
            unhydrated_response = self.response_class.model_validate_json(
                cleaned_content
            )
            processed_response = self.rec_expansion_fn(unhydrated_response, messages)
            return {"messages": [AIMessage(content=processed_response)]}
        except Exception as e:
            self.logger.error(f"Error processing final response: {e}")
            # Return the original response if processing fails
            return {"messages": [response]}


class ShopAgent:
    """Shop agent to provide shop recommendations."""

    def __init__(
        self,
        model: BaseChatModel,
        tools: Sequence[BaseTool],
        system_message: BaseMessage | List[BaseMessage],
        response_class: Type[BaseModel],
        hydrated_response_class: Type[BaseModel],
        rec_expansion_fn: Callable,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        """Initialize the stylist agent.

        Args:
            model: The model to use for the agent.
            tools: The tools to use for the agent.
            system_message: The system message to use for the agent.
            response_class: The response class to use for the agent.
            rec_expansion_fn: The function to use to expand the recommendations. It should have signature `rec_expansion_fn(response: <response_class>) -> str`, where the output string is the jsonified version of the hydrated response class.
            checkpointer: The checkpointer to use for the agent. Defaults to an in-memory checkpointer.
        """
        self.model = model
        self.tools = tools
        self.system_message = system_message
        self.response_class = response_class
        self.rec_expansion_fn = rec_expansion_fn
        logger.info(f"checkpointer: {checkpointer}")
        self.checkpointer = checkpointer or InMemorySaver()
        self.agent_executor = self.compile_agent_graph()
        self.hydrated_response_class = hydrated_response_class

        sig = inspect.signature(self.rec_expansion_fn)
        assert len(sig.parameters) == 2, (
            "rec_expansion_fn must take exactly 2 arguments"
        )

    def compile_agent_graph(self) -> CompiledStateGraph:
        """Compile the agent graph."""
        workflow = StateGraph(ShopAgentState)

        # Create nodes
        model_node = ShopAgentModelNode(
            self.model,
            self.tools or [],
            self.system_message,
            self.response_class,
            self.rec_expansion_fn,
        )
        workflow.add_node("model", model_node)
        workflow.add_node("tools", ToolNode(self.tools or []))

        # Set up the workflow
        workflow.set_entry_point("model")
        workflow.add_conditional_edges(
            "model",
            self.should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )
        workflow.add_edge("tools", "model")

        # Compile the graph
        return workflow.compile(checkpointer=self.checkpointer)

    @staticmethod
    def should_continue(state: ShopAgentState) -> Literal["continue", "end"]:
        """Determine if the agent should continue or end."""
        messages = state["messages"]
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        else:
            return "end"

    async def run(self, message: str, config: Optional[ShopAgentConfig] = None) -> str:
        """Run the agent with the given message."""
        try:
            logger.debug("Starting agent", message=message)

            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "is_last_step": False,
            }

            # Prepare config for runnable
            config_dict = {}
            if config:
                config_dict = config.model_dump()

            # Run the agent
            response = await self.agent_executor.ainvoke(
                initial_state,
                RunnableConfig(configurable=config_dict),
            )

            # Process the final response
            messages = response.get("messages", [])
            if messages:
                final_message = messages[-1]
                if hasattr(final_message, "content"):
                    if isinstance(final_message.content, str):
                        return final_message.content
                    else:
                        return str(final_message.content)

            return json.dumps({"error": "No response from agent"})
        except Exception as e:
            logger.error("Error running stylist agent", error=e)
            return json.dumps({"error": str(e)})

    async def stream(
        self,
        message: str,
        config: ShopAgentConfig,
    ) -> AsyncIterator[dict]:
        tool_messages: List[ToolMessage] = []
        async for c in self.checkpointer.alist(
            RunnableConfig(configurable=config.model_dump())
        ):
            channel_values = c.checkpoint.get("channel_values", {})
            if "messages" in channel_values:
                for tool_message in channel_values["messages"]:
                    if isinstance(tool_message, ToolMessage):
                        tool_messages.append(tool_message)

        logger.info(
            "Finding tool messages from checkpointer",
            num_tool_messages=len(tool_messages),
        )
        logger.debug("tool messages", tool_messages=tool_messages)
        response_parser = JsonOutputParser(pydantic_object=self.response_class)
        try:
            logger.info(
                "Starting stream",
                message=message,
                thread_id=config.thread_id,
            )

            send_metadata = False
            metadata = {
                "run_id": config.run_id,
            }
            steps = 0

            # Initialize the state with both user_id and user_features
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "is_last_step": False,
            }

            logger.debug("Created initial state", state=initial_state)

            # Keep track of the current message state
            current_parsed_message = None
            # Current AI message content being built
            current_ai_content = ""
            # Flag to indicate if we are receiving chunks from the LLM
            receiving_llm_chunks = False
            # Last time we sent an update to the client
            last_update_time = 0
            # Minimum milliseconds between updates to avoid overwhelming the client
            MIN_UPDATE_INTERVAL = 10  # milliseconds
            # Create dynamic state that will be updated with tool messages
            current_state = dict(initial_state)

            # Define stream modes as literal types
            stream_modes: List[
                Literal["updates", "messages", "values", "debug", "custom"]
            ] = ["updates", "messages"]

            if self.agent_executor is None:
                raise ValueError("Agent executor not initialized")
            async for chunk in self.agent_executor.astream(
                initial_state,
                RunnableConfig(
                    configurable=config.model_dump(),
                ),
                stream_mode=stream_modes,
            ):
                # Send metadata once at the beginning
                if not send_metadata:
                    send_metadata = True
                    yield {"event": "metadata", "data": json.dumps(metadata)}

                # Handle progress updates
                if isinstance(chunk, dict) and "updates" in chunk:
                    steps += 1
                    progress = {"steps": steps}
                    logger.info("Progress update", progress=progress)
                    yield {"event": "data", "data": json.dumps(progress)}
                    continue

                # Check if this is a tuple containing message updates
                if isinstance(chunk, tuple) and len(chunk) >= 2:
                    # Extract all messages from the chunk
                    state_messages = chunk[1]
                    if state_messages:
                        # Update the current state with all messages
                        current_state["messages"] = list(state_messages)

                        # Extract tool messages
                        for msg in state_messages:
                            if isinstance(msg, ToolMessage):
                                # Add to tool messages list if not already there
                                if msg not in tool_messages:
                                    tool_messages.append(msg)
                                    logger.debug(f"Captured tool message: {msg.name}")

                # Skip chunks that don't contain AI message content
                if not isinstance(chunk, tuple) or len(chunk) < 2:
                    continue

                # Extract message content from the chunk
                try:
                    message_chunk = chunk[1][0]  # Access the message in the chunk

                    # Skip tool messages and non-AI messages
                    if not isinstance(message_chunk, AIMessage):
                        continue

                    # Skip chunks with tool calls
                    if (
                        hasattr(message_chunk, "tool_calls")
                        and message_chunk.tool_calls
                    ):
                        continue

                    # Get the content from the message chunk
                    chunk_content = message_chunk.content
                    if not chunk_content:
                        continue

                    # Start tracking LLM chunks
                    receiving_llm_chunks = True

                    # Append to our accumulated content
                    if isinstance(chunk_content, str):
                        current_ai_content += chunk_content
                    elif isinstance(chunk_content, list):
                        # Join list items or convert them to string
                        current_ai_content += " ".join(
                            str(item) for item in chunk_content
                        )
                    else:
                        # For any other type, convert to string
                        current_ai_content += str(chunk_content)

                    # Rate limit updates to avoid overwhelming the client
                    current_time = int(time.time() * 1000)
                    if current_time - last_update_time < MIN_UPDATE_INTERVAL:
                        continue

                    # Try to parse the accumulated content
                    try:
                        # Clean the content (remove markdown code blocks if present)
                        cleaned_content = (
                            current_ai_content.replace("```json", "")
                            .replace("```", "")
                            .strip()
                        )

                        # Try to parse with agent_response_parser
                        parsed_message = response_parser.parse(cleaned_content)

                        # Skip if parsing failed or isn't a proper message format
                        if not isinstance(parsed_message, dict):
                            continue

                        # Create a state with the current messages including tool messages
                        rendering_state = dict(current_state)

                        # IMPORTANT: Ensure tool messages are included in the state
                        # This ensures they'll be available for product rendering
                        if tool_messages:
                            logger.debug("we have tool messages")
                            # Create a combined message list that includes all messages
                            messages_from_state = rendering_state.get("messages", [])
                            combined_messages = []
                            # First add existing messages
                            if isinstance(messages_from_state, list):
                                combined_messages.extend(messages_from_state)
                            # Add any tool messages that aren't already in the state
                            for tm in tool_messages:
                                if tm not in combined_messages:
                                    combined_messages.append(tm)
                            # Update the state with the combined messages
                            rendering_state["messages"] = combined_messages
                            # Parse the rendered content back to a dict for sanity check
                            try:
                                # Only use rec_expansion_fn for recommendations or when tool messages exist
                                # Render products with justification using the updated state
                                unhydrated_obj = self.response_class.model_validate(
                                    parsed_message
                                )
                                logger.debug(
                                    "attempting to expand",
                                    unhydrated_obj=unhydrated_obj,
                                )
                                rendered_content = self.rec_expansion_fn(
                                    unhydrated_obj, combined_messages
                                )
                                hydrated_obj = (
                                    self.hydrated_response_class.model_validate_json(
                                        rendered_content
                                    )
                                )
                                parsed_message = hydrated_obj.model_dump()
                            except Exception as e:
                                # Log the error but continue to collect chunks
                                logger.debug(
                                    f"Unable to parse rendered content to agent response type: {parsed_message}, continuing to collect chunks, error: {e}"
                                )
                                # traceback.print_exc()
                                continue

                        # Create patch between previous and current message
                        if current_parsed_message is None:
                            # First message, sanitize and send the full message
                            patch = parsed_message
                        else:
                            # For subsequent messages, create a patch
                            try:
                                # Sanitize the message before creating a patch
                                json_patch = jsonpatch.make_patch(
                                    current_parsed_message,
                                    parsed_message,
                                )
                                patch = json_patch.patch
                            except Exception as e:
                                logger.warning(f"Failed to create patch: {e}")
                                # Fall back to sending the full message if patch creation fails
                                patch = parsed_message
                        yield {
                            "event": "data",
                            "data": json.dumps(patch),
                        }

                        # Update the current message for the next comparison
                        if isinstance(patch, dict):
                            current_parsed_message = patch
                        else:
                            # If we generated a patch object, apply it to current_parsed_message
                            current_parsed_message = jsonpatch.apply_patch(
                                current_parsed_message, patch
                            )

                        # Update last update time
                        last_update_time = current_time

                    except Exception as parse_error:
                        # Log but continue collecting chunks if parsing fails
                        logger.debug(
                            f"Error parsing accumulated content: {parse_error}"
                        )
                        content_sample = (
                            current_ai_content[:100] + "..."
                            if len(current_ai_content) > 100
                            else current_ai_content
                        )
                        logger.debug(f"Content sample: {content_sample}")
                        continue

                except Exception as e:
                    logger.debug(f"Error processing message chunk: {e}")
                    if isinstance(chunk, tuple) and len(chunk) > 1:
                        logger.debug(f"Chunk message type: {type(chunk[1])}")
                    continue

            # If we received LLM chunks but couldn't parse a message, send a basic response
            if receiving_llm_chunks and current_parsed_message is None:
                # Send a waiting event to the client
                yield {
                    "event": "waiting",
                }

            # Send end event
            logger.info("Stream complete", thread_id=config.thread_id)
            yield {"event": "end"}

        except Exception as e:
            logger.error(f"Error in stream: {e}")
            traceback.print_exc()

            error_message = f"Internal Server Error: {str(e)}"

            yield {
                "event": "error",
                "data": json.dumps({"status_code": 500, "message": error_message}),
            }
            raise
