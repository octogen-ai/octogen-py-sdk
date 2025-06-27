import json
from datetime import datetime
from typing import List, Optional

import structlog
from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import CheckpointTuple

from octogen.shop_agent.checkpointer import ShopAgentInMemoryCheckpointSaver
from octogen.shop_agent.schemas import (
    ChatHistory,
    ChatMessage,
    HydratedAgentResponse,
    ProductRecommendation,
    Thread,
)

logger = structlog.get_logger()


async def list_threads_for_user(user_id: str, checkpointer: Optional[ShopAgentInMemoryCheckpointSaver] = None) -> List[Thread]:
    """
    Lists all conversation threads for a given user.

    Args:
        user_id: The ID of the user.

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
        try:
            # Attempt to get the title from the first message
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
    logger.info(f"Found {len(threads)} threads for user {user_id}")
    return threads


async def get_chat_history_for_thread(*, user_id: str, thread_id: str, checkpointer: Optional[ShopAgentInMemoryCheckpointSaver] = None) -> ChatHistory:
    """
    Retrieves the chat history for a specific thread and user.

    Args:
        user_id: The ID of the user.
        thread_id: The ID of the thread.

    Returns:
        A ChatHistory object containing all messages and metadata for the thread.
    """
    thread_checkpoints = []
    # Collect all conversation messages for the given thread and user
    async for checkpoint in checkpointer.afind_conversation_messages(
        user_id=user_id, thread_id=thread_id
    ):
        thread_checkpoints.append(checkpoint)
    logger.info(f"Found {len(thread_checkpoints)} checkpoints for thread {thread_id}")
    # Process the collected checkpoints to build the chat history
    return get_chat_history_from_checkpoint_tuples(thread_checkpoints)


def get_chat_history_from_checkpoint_tuples(
    checkpoint_tuples: List[CheckpointTuple],
) -> ChatHistory:
    """
    Constructs a ChatHistory object from a list of checkpoint tuples.

    Args:
        checkpoint_tuples: A list of checkpoint tuples from a conversation.

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
                        json_content = json.loads(content)
                        structured_response = HydratedAgentResponse(
                            **json_content,
                            product_recommendations=[
                                ProductRecommendation(**rec)
                                for rec in json_content.get(
                                    "product_recommendations", []
                                )
                            ],
                        )
                        messages.append(
                            ChatMessage(
                                timestamp=timestamp,
                                role="assistant",
                                content=structured_response,
                            )
                        )
                    except json.JSONDecodeError:
                        # Fallback for malformed JSON
                        logger.debug(f"Failed to parse content as JSON: {content}")
                # Handle simple string content
                elif isinstance(content, str):
                    messages.append(
                        ChatMessage(
                            timestamp=timestamp,
                            role="assistant",
                            content=HydratedAgentResponse(
                                response_type="freeform_question", preamble=content
                            ),
                        )
                    )

    # Filter out empty messages
    messages = [msg for msg in messages if msg.content]

    if not messages:
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
    title = (
        messages[0].content[:50]
        if messages and isinstance(messages[0].content, str)
        else "Conversation"
    )

    return ChatHistory(
        messages=messages, thread_id=thread_id, created_at=created_at, title=str(title)
    )


async def delete_thread(user_id: str, thread_id: str, checkpointer: Optional[ShopAgentInMemoryCheckpointSaver] = None) -> int:
    """
    Deletes all checkpoints for a given thread.

    Args:
        user_id: The ID of the user (for consistency, not used in this implementation).
        thread_id: The ID of the thread to delete.

    Returns:
        The number of checkpoints deleted.
    """
    # The user_id is not strictly necessary for deletion in this in-memory implementation
    # but is kept for API consistency.
    if not checkpointer:
        checkpointer = ShopAgentInMemoryCheckpointSaver()
    deleted_count = await checkpointer.adelete_thread_checkpoints(thread_id)
    return deleted_count
