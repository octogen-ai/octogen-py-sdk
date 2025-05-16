# Octogen Python SDK for LangGraph

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
    tool_names=["search_products", "get_product_details"],
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
