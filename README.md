# Octogen Python SDK built on LangGraph

[![PyPI version](https://badge.fury.io/py/octogen-sdk-langgraph.svg)](https://badge.fury.io/py/octogen-sdk-langgraph)
[![Python Version](https://img.shields.io/pypi/pyversions/octogen-sdk-langgraph.svg)](https://pypi.org/project/octogen-sdk-langgraph/)

A Python SDK for building LLM-powered shop agents using LangGraph and LangChain, designed to work with the Octogen platform.

## Features

- Build conversational shop agents with LangGraph's state management
- Structured output parsing with Pydantic models
- Built-in recommendation expansion functionality
- Server deployment capabilities with FastAPI
- Integration with Octogen MCP tools for product discovery
- Streamlined agent creation through factory patterns

## Environment Variables

### Required Variables
- `OPENAI_API_KEY` - Your OpenAI API key
- `OCTOGEN_API_KEY` - Your Octogen API key
- `OCTOGEN_MCP_SERVER_HOST` - Octogen MCP server host URL

### Optional Variables (for LangChain Tracing)
- `LANGCHAIN_API_KEY` - Your LangChain API key
- `LANGCHAIN_TRACING_V2` - Enable LangChain tracing (set to "true")
- `LANGCHAIN_PROJECT` - LangChain project name

### .env File Placement
For the SDK to properly load your environment variables, you can:

1. **Place a .env file in your project's root directory** - The default behavior is to look for a .env file in the current working directory.

2. **Explicitly specify the path** - When using example servers or creating agents, pass the path to your .env file:
   ```python
   from dotenv import find_dotenv
   from octogen.shop_agent.settings import get_agent_settings

   # Pass the path to your .env file
   get_agent_settings(find_dotenv(usecwd=True))
   ```

3. **Set environment variables directly** - You can also set these variables in your environment before running your application.

### Example Projects
When running the example projects (stylist, discovery, comparison), place your `.env` file in the specific example's directory. For instance, to run the stylist example:

```
examples/stylist/.env  # Place your .env file here when running the stylist example
```

This is because the examples use `find_dotenv(usecwd=True)`, which looks for a `.env` file in the current working directory.

## Installation

```bash
pip install octogen-sdk-langgraph
```

## Requirements

- Python ≥ 3.12
- Dependencies:
  - langchain ≥ 0.3.25
  - langgraph ≥ 0.4.3
  - pydantic ≥ 2.11.4
  - octogen-api ≥ 0.1.0a4
  - structlog ≥ 25.3.0

## Quick Start

```python
from langchain_openai import ChatOpenAI
from octogen.shop_agent import ShopAgent, create_agent
from your_models import ResponseClass, HydratedResponseClass

# Define your recommendation expansion function
def expand_recommendations(response, messages):
    # Process and expand recommendations
    return json.dumps(expanded_response)

# Create a shop agent
async with create_agent(
    model=ChatOpenAI(model="gpt-4"),
    agent_name="MyShopAgent",
    response_class=ResponseClass,
    hydrated_response_class=HydratedResponseClass,
    rec_expansion_fn=expand_recommendations,
    tool_names=["agent_search_products", "enrich_product_image"],
    hub_prompt_id="your/hub/prompt_id",
) as agent:
    # Use the agent
    result = await agent.run("I'm looking for a new jacket")
```

## Usage Examples

See the `examples/` directory for complete implementations:
- `examples/stylist/` - A personal shopping assistant
- `examples/discovery/` - Product discovery agent
- `examples/comparison/` - Product comparison tool

## License

This project is licensed under the terms included in the LICENSE file.
