from dotenv import find_dotenv
from langchain_openai import ChatOpenAI
from showcase.agent import create_comparison_agent
from showcase.schema import HydratedComparisonResponse

from octogen.shop_agent.server import AgentServer
from octogen.shop_agent.settings import get_agent_settings


def run_server(host: str = "0.0.0.0", port: int = 8003) -> None:
    """Run the comparison agent server."""
    # Initialize settings
    get_agent_settings(find_dotenv(usecwd=True))

    # Create server
    server = AgentServer(
        title="Comparison Agent",
        endpoint_path="comparison",
        agent_factory=lambda: create_comparison_agent(
            model=ChatOpenAI(model="gpt-4.1")
        ),
        response_model=HydratedComparisonResponse,
    )

    # Run server
    server.run(host=host, port=port)


if __name__ == "__main__":
    run_server()
