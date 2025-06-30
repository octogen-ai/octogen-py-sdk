import os
from functools import lru_cache
from typing import Optional

import structlog
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

logger = structlog.get_logger(__name__)

DEFAULT_MCP_SERVER_TIMEOUT = 3600


class AgentSettings(BaseSettings):
    mcp_api_key: str
    mcp_server_host: str
    mcp_server_url: str
    mcp_auth_header: dict[str, str]
    mcp_server_timeout: int = DEFAULT_MCP_SERVER_TIMEOUT


@lru_cache
def get_agent_settings(path: Optional[str] = None) -> AgentSettings:
    if path:
        load_dotenv(path)
    else:
        load_dotenv()
    if not (
        os.getenv("LANGCHAIN_API_KEY")
        and os.getenv("LANGCHAIN_TRACING_V2")
        and os.getenv("LANGCHAIN_PROJECT")
    ):
        logger.warning(
            "Missing environment variables for LangChain tracing", env=os.environ
        )

    if not (os.getenv("OPENAI_API_KEY")):
        raise ValueError("OPENAI_API_KEY must be set in the environment.")
    mcp_api_key = os.getenv("OCTOGEN_API_KEY")
    if not isinstance(mcp_api_key, str):
        raise ValueError("OCTOGEN_API_KEY must be set in the environment.")
    mcp_server_host = os.getenv("OCTOGEN_MCP_SERVER_HOST")
    if not mcp_server_host:
        raise ValueError("OCTOGEN_MCP_SERVER_HOST must be set in the environment.")
    # Assume streamable HTTP endpoint at /stream
    mcp_server_url = f"{mcp_server_host.rstrip('/')}/mcp"

    return AgentSettings(
        mcp_api_key=mcp_api_key,
        mcp_server_host=mcp_server_host,
        mcp_server_url=mcp_server_url,
        mcp_auth_header={"x-api-key": mcp_api_key},
        mcp_server_timeout=DEFAULT_MCP_SERVER_TIMEOUT,
    )
