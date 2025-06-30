from typing import Annotated, List

from dotenv import find_dotenv, load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request
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


def get_checkpointer(request: Request) -> ShopAgentInMemoryCheckpointSaver:
    return request.app.state.checkpointer


def run_server(host: str = "0.0.0.0", port: int = 8004) -> None:
    """Run the feed agent server."""
    # Load environment variables but don't validate MCP settings
    load_dotenv(find_dotenv(usecwd=True))

    # Create server and attach checkpointer to its state
    server = AgentServer(
        title="Feed Agent",
        endpoint_path="feed",
        response_model=AgentResponse,
    )
    server.app.state.checkpointer = ShopAgentInMemoryCheckpointSaver()

    # Define the agent factory using the server's checkpointer
    def agent_factory():
        """Factory function returning a configured feed agent."""
        return create_feed_agent(
            model=ChatOpenAI(model="gpt-4.1"),
            checkpointer=server.app.state.checkpointer,
        )

    server.set_agent_factory(agent_factory)

    # Create router for chat history endpoints
    history_router = APIRouter(prefix="/history", tags=["history"])

    CheckpointerDep = Annotated[
        ShopAgentInMemoryCheckpointSaver, Depends(get_checkpointer)
    ]

    @history_router.get("/threads/{user_id}", response_model=List[Thread])
    async def get_threads(user_id: str, checkpointer: CheckpointerDep):
        """List all conversation threads for a user."""
        return await list_threads_for_user(user_id, checkpointer=checkpointer)

    @history_router.get("/threads/{user_id}/{thread_id}", response_model=ChatHistory)
    async def get_chat_history(
        user_id: str, thread_id: str, checkpointer: CheckpointerDep
    ):
        """Get full chat history for a specific thread."""
        return await get_chat_history_for_thread(
            user_id=user_id,
            thread_id=thread_id,
            checkpointer=checkpointer,
            response_model_class=AgentResponse,
        )

    @history_router.delete("/threads/{user_id}/{thread_id}")
    async def remove_thread(
        user_id: str, thread_id: str, checkpointer: CheckpointerDep
    ):
        """Delete a conversation thread."""
        deleted_count = await delete_thread(
            user_id, thread_id, checkpointer=checkpointer
        )
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Thread not found")
        return {"deleted": True, "count": deleted_count}

    # Manually include the history router
    server.app.include_router(history_router)

    # Run server
    server.run(host=host, port=port)


if __name__ == "__main__":
    run_server()
