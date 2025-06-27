from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Dict, List, Tuple

from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runnables import RunnableConfig


class ShopAgentInMemoryCheckpointSaver(InMemorySaver):
    """
    An in-memory checkpoint saver that extends the base InMemorySaver to support
    operations required by the shop agent, such as finding thread boundaries and
    conversation messages for a specific user.
    """

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        """
        Saves a checkpoint by appending it to the list for the given thread_id,
        rather than overwriting.
        """
        thread_id = config["configurable"]["thread_id"]
        self.checkpoints[thread_id].append(
            CheckpointTuple(
                config,
                self.serde.loads(self.serde.dumps(checkpoint)),
                self.serde.loads(self.serde.dumps(metadata)),
                None,
                {},
            )
        )
        return config

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
        # Group checkpoints by thread_id
        threads: Dict[str, List[CheckpointTuple]] = defaultdict(list)
        for thread_id, checkpoints in self.checkpoints.items():
            for checkpoint in checkpoints:
                # Filter checkpoints by user_id from the configurable section
                if checkpoint.config["configurable"].get("user_id") == user_id:
                    threads[thread_id].append(checkpoint)

        # Yield first and last checkpoints for each thread
        for thread_id, checkpoints in threads.items():
            if checkpoints:
                # Sort checkpoints by timestamp to find the first and last
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
        # Retrieve all checkpoints for the given thread_id
        if thread_id in self.checkpoints:
            # Filter by user_id and sort by timestamp
            user_checkpoints = [
                cp
                for cp in self.checkpoints[thread_id]
                if cp.config["configurable"].get("user_id") == user_id
            ]
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
        # Remove the thread and return the number of deleted checkpoints
        if thread_id in self.checkpoints:
            count = len(self.checkpoints[thread_id])
            del self.checkpoints[thread_id]
            return count
        return 0
