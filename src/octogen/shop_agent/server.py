import traceback
import uuid
from contextlib import asynccontextmanager
from typing import (
    AsyncContextManager,
    AsyncGenerator,
    Callable,
    Generic,
    Optional,
    Type,
    TypeVar,
)

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from octogen.shop_agent.base import ShopAgent, ShopAgentConfig

logger = structlog.get_logger()

# Type variables for response models
ResponseT = TypeVar("ResponseT", bound=BaseModel)


# Request model
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    thread_id: Optional[str] = None


class AgentServer(Generic[ResponseT]):
    """Generic server wrapper for shop agents."""

    def __init__(
        self,
        title: str,
        endpoint_path: str,
        response_model: Type[ResponseT],
    ):
        """Initialize a new agent server.

        Args:
            title: The title of the FastAPI application
            endpoint_path: The path for the main endpoint
            response_model: Pydantic model for response validation
        """
        self.title = title
        self.endpoint_path = endpoint_path
        self.agent_factory: Optional[Callable[..., AsyncContextManager[ShopAgent]]] = (
            None
        )
        self.response_model = response_model
        self.agent: Optional[ShopAgent] = None
        self.app = self._create_app()

    def set_agent_factory(
        self, agent_factory: Callable[..., AsyncContextManager[ShopAgent]]
    ) -> None:
        """Set the agent factory after the server is initialized."""
        self.agent_factory = agent_factory

    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        app = FastAPI(title=self.title, lifespan=self._lifespan)

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # For development; restrict in production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Add the main endpoint
        @app.post(
            f"/{self.endpoint_path}",
            response_model=self.response_model,
            operation_id=f"run{self.title.replace(' ', '')}",
        )
        async def process_request(request: ChatRequest) -> ResponseT:
            """Process a request and return a response from the agent."""
            try:
                logger.info(f"Received request: {request.message}")

                # Make sure the agent is initialized
                if self.agent is None:
                    raise HTTPException(
                        status_code=500, detail=f"{self.title} agent not initialized"
                    )
                logger.info(f"Agent initialized with user_id: {request.user_id}")
                logger.info(f"Agent initialized with thread_id: {request.thread_id}")

                # Determine / generate a title for this thread
                title = await self._get_or_generate_title(
                    request=request,
                )

                # Configure the agent, including the title so it is persisted in checkpoints
                config = ShopAgentConfig(
                    user_id=request.user_id or "",
                    thread_id=request.thread_id or "",
                    run_id=str(uuid.uuid4()),
                    title=title,
                )

                # Process the message with the agent
                agent_response = await self.agent.run(request.message, config)
                logger.info(f"agent checkpoints: {self.agent.checkpointer.storage}")
                try:
                    response = self.response_model.model_validate_json(agent_response)
                except Exception as e:
                    logger.error(f"Error validating response: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

                # Log a sample of the response
                logger.info(f"Generated response: {response}")

                return response
            except Exception as e:
                logger.error(f"Error processing chat request: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        return app

    @asynccontextmanager
    async def _lifespan(self, _: FastAPI) -> AsyncGenerator[None, None]:
        """Lifespan context manager for FastAPI application lifecycle events."""
        # Startup: initialize the agent
        logger.info(f"Initializing {self.title} agent...")
        if not self.agent_factory:
            raise RuntimeError("Agent factory not set.")
        try:
            async with self.agent_factory() as agent:
                self.agent = agent
                logger.info(f"{self.title} agent initialized successfully")
                yield  # FastAPI takes over here to handle requests
        except Exception as e:
            logger.error(f"Failed to initialize {self.title} agent: {e}")
            logger.error(traceback.format_exc())

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Run the agent server."""
        uvicorn.run(self.app, host=host, port=port)

    # Helper -------------------------------------------------------------

    async def _get_or_generate_title(self, request: "ChatRequest") -> str:  # type: ignore[name-defined]
        """Return existing title from checkpoints or produce a new one via LLM."""

        # If we already have an agent/checkpointer
        if self.agent and request.thread_id:
            try:
                async for (
                    thread_id,
                    first_cp,
                    _last_cp,
                ) in self.agent.checkpointer.afind_thread_boundary_checkpoints(
                    request.user_id or ""
                ):
                    if thread_id == request.thread_id:
                        cfg = first_cp.config["configurable"]
                        existing_title = cfg.get("title")
                        if existing_title:
                            return existing_title
            except Exception as e:
                logger.debug(f"Error fetching existing title: {e}")

        # No existing title — generate one using LLM
        try:
            model = ChatOpenAI(model="gpt-4.1")
            prompt = (
                "You are an assistant that writes short, descriptive titles. "
                "Summarise the following user message into a concise conversation title of at most 6 words.\n\n"
                f"Message:\n{request.message}\n\nTitle:"
            )
            title: str = await model.apredict(prompt)  # type: ignore[attr-defined]
            return title.strip().strip('"')
        except Exception as e:
            logger.warning(f"Failed to generate title with LLM: {e}")
            # Fallback to first 50 chars of message
            return (request.message[:50] if request.message else "Conversation").strip()
