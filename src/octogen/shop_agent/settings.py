import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    octogen_api_key: str
    octogen_mcp_server_host: str


@lru_cache
def get_settings(path: Optional[str] = None) -> Settings:
    if path:
        load_dotenv(path)
    else:
        load_dotenv()
    if not (
        os.getenv("LANGCHAIN_API_KEY")
        and os.getenv("LANGCHAIN_TRACING_V2")
        and os.getenv("LANGCHAIN_PROJECT")
        and os.getenv("OPENAI_API_KEY")
        and os.getenv("OCTOGEN_API_KEY")
    ):
        raise ValueError(
            "LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2, LANGCHAIN_PROJECT, OPENAI_API_KEY and OCTOGEN_API_KEY must be set in the environment."
        )

    octogen_api_key = os.getenv("OCTOGEN_API_KEY")
    if not isinstance(octogen_api_key, str):
        raise ValueError("OCTOGEN_API_KEY must be set in the environment.")
    octogen_mcp_server_host = os.getenv("OCTOGEN_MCP_SERVER_HOST")
    if not octogen_mcp_server_host:
        raise ValueError("OCTOGEN_MCP_SERVER_HOST must be set in the environment.")
    return Settings(
        octogen_api_key=octogen_api_key,
        octogen_mcp_server_host=octogen_mcp_server_host,
    )


class MCPSettings(BaseModel):
    mcp_server_url: str
    auth_header: dict[str, str]


@lru_cache
def get_mcp_settings() -> MCPSettings:
    return MCPSettings(
        mcp_server_url=f"{get_settings().octogen_mcp_server_host}/sse",
        auth_header={"x-api-key": get_settings().octogen_api_key},
    )
