import traceback
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import structlog
import uvicorn
from dotenv import find_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from showcase.agent import create_discovery_agent
from showcase.schema import AgentResponse

from octogen.shop_agent.base import ShopAgent, ShopAgentConfig
from octogen.shop_agent.settings import get_settings

logger = structlog.get_logger()

# Global agent instance
_discovery_agent: Optional[ShopAgent] = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for FastAPI application lifecycle events."""
    # Startup: initialize the comparison agent
    logger.info("Initializing comparison agent...")
    get_settings(find_dotenv(usecwd=True))
    try:
        async with create_discovery_agent(model=ChatOpenAI(model="gpt-4.1")) as agent:
            global _discovery_agent
            _discovery_agent = agent
            logger.info("Discovery agent initialized successfully")
            yield  # FastAPI takes over here to handle requests
    except Exception as e:
        logger.error(f"Failed to initialize discovery agent: {e}")
        logger.error(traceback.format_exc())

    # Shutdown: No cleanup needed for now


app = FastAPI(title="Discovery Agent API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    thread_id: Optional[str] = None


@app.post(
    "/discovery",
    response_model=AgentResponse,
    operation_id="runDiscoveryAgent",
)
async def discovery(request: ChatRequest) -> AgentResponse:
    """Process a discovery request and return a response from the discovery agent."""
    global _discovery_agent

    try:
        logger.info(f"Received comparison request: {request.message}")

        # Make sure the agent is initialized
        if _discovery_agent is None:
            raise HTTPException(
                status_code=500, detail="Discovery agent not initialized"
            )

        # Configure the agent
        config = ShopAgentConfig(
            user_id=request.user_id or "",
            thread_id=request.thread_id or "",
            run_id=str(uuid.uuid4()),  # Generate a unique ID if needed
        )

        # Process the message with the agent
        agent_response = await _discovery_agent.run(request.message, config)
        try:
            response = AgentResponse.model_validate_json(agent_response)
        except Exception as e:
            logger.error(f"Error validating response: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        # Log a sample of the response
        logger.info(f"Generated response: {response}")

        return response
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "0.0.0.0", port: int = 8004) -> None:
    """Run the discovery agent server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
