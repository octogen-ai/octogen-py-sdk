from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver

logger = structlog.get_logger()


class ShopAgentInMemoryCheckpointSaver(InMemorySaver):
    """
    An in-memory checkpoint saver that extends the base InMemorySaver to support
    operations required by the shop agent, such as finding thread boundaries and
    conversation messages for a specific user.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Store full configs separately
        self.configs: Dict[Tuple[str, str, str], RunnableConfig] = {}

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Override to store the full config including user_id."""
        # Call parent's put method (store the minimal checkpoint via parent logic)
        super().put(config, checkpoint, metadata, new_versions)

        # Store the full config
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        self.configs[(thread_id, checkpoint_ns, checkpoint_id)] = config

        # Return the full config instead of just the minimal one
        return config

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Override to return the full config including user_id."""
        # Get the base tuple from parent
        tuple_result = super().get_tuple(config)
        if not tuple_result:
            return None

        # Get the stored full config
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = tuple_result.checkpoint["id"]

        full_config = self.configs.get((thread_id, checkpoint_ns, checkpoint_id))
        if full_config:
            # Replace the config with the full one
            return CheckpointTuple(
                config=full_config,
                checkpoint=tuple_result.checkpoint,
                metadata=tuple_result.metadata,
                parent_config=tuple_result.parent_config,
                pending_writes=tuple_result.pending_writes,
            )
        return tuple_result

    async def afind_thread_boundary_checkpoints(
        self, user_id: str
    ) -> AsyncIterator[Tuple[str, CheckpointTuple, CheckpointTuple]]:
        """
        Finds the first and last checkpoints for each thread associated with a user.

        Args:
            user_id: The ID of the user whose threads are to be found.

        Yields:
            An iterator of tuples, each containing the thread_id, the first checkpoint,
            and the last checkpoint of a thread.
        """
        user_threads: DefaultDict[str, List[CheckpointTuple]] = defaultdict(lambda: [])

        logger.info(f"Looking for threads for user_id: {user_id}")
        logger.info(f"Storage contains {len(self.storage)} threads")

        for thread_id, namespaces in self.storage.items():
            logger.info(f"Checking thread_id: {thread_id}")
            for checkpoint_ns, checkpoint_ids in namespaces.items():
                logger.info(
                    f"  Namespace: {checkpoint_ns}, checkpoints: {len(checkpoint_ids)}"
                )
                for checkpoint_id in checkpoint_ids:
                    config: RunnableConfig = {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        }
                    }
                    cp_tuple = self.get_tuple(config)
                    if cp_tuple:
                        logger.info(f"    Checkpoint config: {cp_tuple.config}")
                        config_user_id = cp_tuple.config["configurable"].get("user_id")
                        logger.info(
                            f"    Config user_id: {config_user_id}, looking for: {user_id}"
                        )
                        if config_user_id == user_id:
                            user_threads[thread_id].append(cp_tuple)
                        else:
                            logger.info("    User ID mismatch")

        logger.info(f"Found {len(user_threads)} threads for user {user_id}")
        for thread_id, checkpoints in user_threads.items():
            if checkpoints:
                checkpoints.sort(key=lambda cp: cp.checkpoint["ts"])
                yield thread_id, checkpoints[0], checkpoints[-1]

    async def afind_conversation_messages(
        self, *, user_id: str, thread_id: str
    ) -> AsyncIterator[CheckpointTuple]:
        """
        Retrieves all conversation messages for a given thread and user, sorted by time.

        Args:
            user_id: The ID of the user.
            thread_id: The ID of the thread.

        Yields:
            An iterator of checkpoints that belong to the specified conversation.
        """
        if thread_id not in self.storage:
            return

        user_checkpoints: List[CheckpointTuple] = []
        for checkpoint_ns, checkpoint_ids in self.storage[thread_id].items():
            for checkpoint_id in checkpoint_ids:
                config: RunnableConfig = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                }
                cp_tuple = self.get_tuple(config)
                if (
                    cp_tuple
                    and cp_tuple.config["configurable"].get("user_id") == user_id
                ):
                    user_checkpoints.append(cp_tuple)

        user_checkpoints.sort(key=lambda cp: cp.checkpoint["ts"])
        for checkpoint in user_checkpoints:
            yield checkpoint

    async def adelete_thread_checkpoints(self, thread_id: str) -> int:
        """
        Deletes all checkpoints associated with a specific thread.

        Args:
            thread_id: The ID of the thread to delete.

        Returns:
            The number of checkpoints deleted.
        """
        count = 0
        if thread_id in self.storage:
            for ns in self.storage[thread_id]:
                for checkpoint_id in self.storage[thread_id][ns]:
                    # Clean up stored configs
                    config_key = (thread_id, ns, checkpoint_id)
                    if config_key in self.configs:
                        del self.configs[config_key]
                count += len(self.storage[thread_id][ns])

            self.delete_thread(thread_id)
        return count
