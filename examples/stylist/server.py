from dotenv import find_dotenv
from langchain_openai import ChatOpenAI
from showcase.agent import create_stylist_agent
from showcase.schema import HydratedStylistAgentResponse

from octogen.shop_agent.server import AgentServer
from octogen.shop_agent.settings import get_agent_settings


def run_server(host: str = "0.0.0.0", port: int = 8005) -> None:
    """Run the stylist agent server."""
    # Initialize settings
    get_agent_settings(find_dotenv(usecwd=True))

    # Create server
    server = AgentServer(
        title="Stylist Agent",
        endpoint_path="stylist",
        agent_factory=lambda: create_stylist_agent(model=ChatOpenAI(model="gpt-4.1")),
        response_model=HydratedStylistAgentResponse,
    )

    # Run server
    server.run(host=host, port=port)


if __name__ == "__main__":
    run_server()
