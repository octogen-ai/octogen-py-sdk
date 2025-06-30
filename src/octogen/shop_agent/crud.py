import json
from datetime import datetime
from typing import List, Type

import structlog
from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import CheckpointTuple
from pydantic import BaseModel, ValidationError

from octogen.shop_agent.checkpointer import ShopAgentInMemoryCheckpointSaver
from octogen.shop_agent.schemas import (
    ChatHistory,
    ChatMessage,
    HydratedAgentResponse,
    Thread,
)

logger = structlog.get_logger()


async def list_threads_for_user(
    user_id: str, checkpointer: ShopAgentInMemoryCheckpointSaver
) -> List[Thread]:
    """
    Lists all conversation threads for a given user.

    Args:
        user_id: The ID of the user.
        checkpointer: The checkpoint saver instance.

    Returns:
        A list of Thread objects, each representing a conversation thread.
    """
    threads = []
    # Use the checkpointer to find the first and last checkpoints for each thread
    async for (
        thread_id,
        first_checkpoint,
        last_checkpoint,
    ) in checkpointer.afind_thread_boundary_checkpoints(user_id):
        # Extract timestamps and title from the checkpoints
        created_at = datetime.fromisoformat(first_checkpoint.checkpoint["ts"])
        updated_at = datetime.fromisoformat(last_checkpoint.checkpoint["ts"])
        # Prefer a title stored in the checkpoint's configurable parameters
        title: str | None = first_checkpoint.config["configurable"].get("title")

        if not title:
            try:
                # Attempt to get the title from the first user message
                title = first_checkpoint.checkpoint["channel_values"]["__start__"][
                    "messages"
                ][0].content
            except (KeyError, IndexError):
                # Fallback title if the expected structure is not present
                title = "Conversation"

        threads.append(
            Thread(
                thread_id=thread_id,
                created_at=created_at,
                updated_at=updated_at,
                title=title,
            )
        )
    logger.info(f"found {len(threads)} threads for user {user_id}")
    return threads


async def get_chat_history_for_thread(
    *,
    user_id: str,
    thread_id: str,
    checkpointer: ShopAgentInMemoryCheckpointSaver,
    response_model_class: Type[BaseModel] = HydratedAgentResponse,
) -> ChatHistory:
    """
    Retrieves the chat history for a specific thread and user.

    Args:
        user_id: The ID of the user.
        thread_id: The ID of the thread.
        checkpointer: The checkpoint saver instance.
        response_model_class: The response model class to use for parsing JSON content.

    Returns:
        A ChatHistory object containing all messages and metadata for the thread.
    """
    thread_checkpoints = []
    # Collect all conversation messages for the given thread and user
    async for checkpoint in checkpointer.afind_conversation_messages(
        user_id=user_id, thread_id=thread_id
    ):
        thread_checkpoints.append(checkpoint)

    # Process the collected checkpoints to build the chat history
    return get_chat_history_from_checkpoint_tuples(
        checkpoint_tuples=thread_checkpoints,
        response_model_class=response_model_class,
    )


def get_chat_history_from_checkpoint_tuples(
    checkpoint_tuples: List[CheckpointTuple],
    *,
    response_model_class: Type[BaseModel] = HydratedAgentResponse,
) -> ChatHistory:
    """
    Constructs a ChatHistory object from a list of checkpoint tuples.

    Args:
        checkpoint_tuples: A list of checkpoint tuples from a conversation.
        response_model_class: The response model class to use for parsing JSON content.

    Returns:
        A ChatHistory object representing the conversation.
    """
    messages = []
    logger.info(f"Processing {len(checkpoint_tuples)} checkpoints")

    for checkpoint_tuple in checkpoint_tuples:
        timestamp = datetime.fromisoformat(checkpoint_tuple.checkpoint["ts"])
        channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})

        # Process the initial user message
        if "__start__" in channel_values:
            base_message = channel_values["__start__"]["messages"][-1]
            if hasattr(base_message, "content"):
                messages.append(
                    ChatMessage(
                        timestamp=timestamp, role="user", content=base_message.content
                    )
                )

        # Process other messages in the conversation
        elif "messages" in channel_values:
            base_message = channel_values["messages"][-1]
            if isinstance(base_message, AIMessage):
                content = base_message.content
                # Handle structured JSON content
                if (
                    isinstance(content, str)
                    and content.startswith("{")
                    and "response_type" in content
                ):
                    try:
                        # Attempt to parse the JSON into the provided response model.
                        structured_response = response_model_class.parse_raw(content)
                        messages.append(
                            ChatMessage(
                                timestamp=timestamp,
                                role="assistant",
                                # Convert the BaseModel to a plain dict so FastAPI/Pydantic
                                # encodes it correctly in the API response.
                                content=structured_response.model_dump(
                                    exclude_none=True
                                ),
                            )
                        )
                    except (ValidationError, json.JSONDecodeError) as e:
                        # If parsing fails, log and fall back to raw string content.
                        logger.debug(
                            f"Failed to parse content as {response_model_class.__name__}: {e}"
                        )
                        messages.append(
                            ChatMessage(
                                timestamp=timestamp,
                                role="assistant",
                                content=content,
                            )
                        )
                # Handle simple string content or fallback when JSON parsing fails.
                elif isinstance(content, str):
                    # For backward compatibility, if the default response model is
                    # being used, wrap plain strings in a HydratedAgentResponse so
                    # downstream consumers continue to receive the structure they
                    # expect. Otherwise, store the raw string.
                    if response_model_class is HydratedAgentResponse:
                        wrapped_content: BaseModel | str | dict = HydratedAgentResponse(
                            response_type="freeform_question",
                            preamble=content,
                        ).model_dump(exclude_none=True)
                    else:
                        wrapped_content = content

                    messages.append(
                        ChatMessage(
                            timestamp=timestamp,
                            role="assistant",
                            content=wrapped_content,
                        )
                    )

    # Filter out empty messages
    messages = [msg for msg in messages if msg.content]

    if not messages:
        if not checkpoint_tuples:
            return ChatHistory(
                messages=[],
                thread_id="",
                created_at=datetime.now(),
                title="Empty conversation",
            )
        # Handle empty conversations
        thread_id = checkpoint_tuples[0].config["configurable"]["thread_id"]
        created_at = datetime.fromisoformat(checkpoint_tuples[0].checkpoint["ts"])
        return ChatHistory(
            messages=[],
            thread_id=thread_id,
            created_at=created_at,
            title="Empty conversation",
        )

    # Build the final chat history object
    thread_id = checkpoint_tuples[0].config["configurable"]["thread_id"]
    created_at = datetime.fromisoformat(checkpoint_tuples[0].checkpoint["ts"])
    # Retrieve title from first checkpoint config if available, otherwise derive from first message
    title: str | None = checkpoint_tuples[0].config["configurable"].get("title")

    if not title:
        title = (
            messages[0].content[:50]
            if messages and isinstance(messages[0].content, str)
            else "Conversation"
        )

    return ChatHistory(
        messages=messages, thread_id=thread_id, created_at=created_at, title=str(title)
    )


async def delete_thread(
    user_id: str, thread_id: str, checkpointer: ShopAgentInMemoryCheckpointSaver
) -> int:
    """
    Deletes all checkpoints for a given thread.

    Args:
        user_id: The ID of the user (for consistency, not used in this implementation).
        thread_id: The ID of the thread to delete.
        checkpointer: The checkpoint saver instance.

    Returns:
        The number of checkpoints deleted.
    """
    # The user_id is not strictly necessary for deletion in this in-memory implementation
    # but is kept for API consistency.
    deleted_count = await checkpointer.adelete_thread_checkpoints(thread_id)
    return deleted_count
