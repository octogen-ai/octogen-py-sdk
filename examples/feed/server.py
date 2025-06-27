from typing import List

from dotenv import find_dotenv, load_dotenv
from fastapi import APIRouter, HTTPException
from langchain_openai import ChatOpenAI
from showcase.agent import create_feed_agent
from showcase.schema import AgentResponse

from octogen.shop_agent.checkpointer import ShopAgentInMemoryCheckpointSaver
from octogen.shop_agent.crud import (
    delete_thread,
    get_chat_history_for_thread,
    list_threads_for_user,
)
from octogen.shop_agent.schemas import ChatHistory, Thread
from octogen.shop_agent.server import AgentServer


def run_server(host: str = "0.0.0.0", port: int = 8004) -> None:
    """Run the feed agent server."""
    # Load environment variables but don't validate MCP settings
    load_dotenv(find_dotenv(usecwd=True))
    checkpointer = ShopAgentInMemoryCheckpointSaver()

    # Create router for chat history endpoints
    history_router = APIRouter(prefix="/history", tags=["history"])

    @history_router.get("/threads/{user_id}", response_model=List[Thread])
    async def get_threads(user_id: str):
        """List all conversation threads for a user."""
        return await list_threads_for_user(user_id, checkpointer=checkpointer)

    @history_router.get("/threads/{user_id}/{thread_id}", response_model=ChatHistory)
    async def get_chat_history(user_id: str, thread_id: str):
        """Get full chat history for a specific thread."""
        return await get_chat_history_for_thread(
            user_id=user_id, thread_id=thread_id, checkpointer=checkpointer
        )

    @history_router.delete("/threads/{user_id}/{thread_id}")
    async def remove_thread(user_id: str, thread_id: str):
        """Delete a conversation thread."""
        deleted_count = await delete_thread(
            user_id, thread_id, checkpointer=checkpointer
        )
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Thread not found")
        return {"deleted": True, "count": deleted_count}

    # Create server
    server = AgentServer(
        title="Feed Agent",
        endpoint_path="feed",
        agent_factory=lambda: create_feed_agent(
            model=ChatOpenAI(model="gpt-4.1"), checkpointer=checkpointer
        ),
        response_model=AgentResponse,
    )

    # Manually include the history router
    server.app.include_router(history_router)

    # Run server
    server.run(host=host, port=port)


if __name__ == "__main__":
    run_server()
