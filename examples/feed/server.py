from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from showcase.agent import create_feed_agent
from showcase.schema import AgentResponse

from octogen.shop_agent.server import AgentServer


def run_server(host: str = "0.0.0.0", port: int = 8004) -> None:
    """Run the feed agent server."""
    # Load environment variables but don't validate MCP settings
    load_dotenv(find_dotenv(usecwd=True))

    # Create server
    server = AgentServer(
        title="Feed Agent",
        endpoint_path="feed",
        agent_factory=lambda: create_feed_agent(model=ChatOpenAI(model="gpt-4.1")),
        response_model=AgentResponse,
    )

    # Run server
    server.run(host=host, port=port)


if __name__ == "__main__":
    run_server()
