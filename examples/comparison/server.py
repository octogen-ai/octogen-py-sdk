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
from showcase.agent import create_comparison_agent
from showcase.schema import HydratedComparisonResponse

from octogen.shop_agent.base import ShopAgent, ShopAgentConfig
from octogen.shop_agent.settings import get_settings

logger = structlog.get_logger()

# Global agent instance
_comparison_agent: Optional[ShopAgent] = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for FastAPI application lifecycle events."""
    # Startup: initialize the comparison agent
    logger.info("Initializing comparison agent...")
    get_settings(find_dotenv(usecwd=True))
    try:
        async with create_comparison_agent(model=ChatOpenAI(model="gpt-4.1")) as agent:
            global _comparison_agent
            _comparison_agent = agent
            logger.info("Comparison agent initialized successfully")
            yield  # FastAPI takes over here to handle requests
    except Exception as e:
        logger.error(f"Failed to initialize comparison agent: {e}")
        logger.error(traceback.format_exc())

    # Shutdown: No cleanup needed for now


app = FastAPI(title="Comparison Agent API", lifespan=lifespan)

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
    "/comparison",
    response_model=HydratedComparisonResponse,
    operation_id="runComparisonAgent",
)
async def comparison(request: ChatRequest) -> HydratedComparisonResponse:
    """Process a comparison request and return a response from the comparison agent."""
    global _stylist_agent

    try:
        logger.info(f"Received comparison request: {request.message}")

        # Make sure the agent is initialized
        if _comparison_agent is None:
            raise HTTPException(
                status_code=500, detail="Comparison agent not initialized"
            )

        # Configure the agent
        config = ShopAgentConfig(
            user_id=request.user_id or "",
            thread_id=request.thread_id or "",
            run_id=str(uuid.uuid4()),  # Generate a unique ID if needed
        )

        # Process the message with the agent
        agent_response = await _comparison_agent.run(request.message, config)
        try:
            response = HydratedComparisonResponse.model_validate_json(agent_response)
        except Exception as e:
            logger.error(f"Error validating response: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        # Log a sample of the response
        logger.info(f"Generated response: {response}")

        return response
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "0.0.0.0", port: int = 8003) -> None:
    """Run the comparison agent server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
