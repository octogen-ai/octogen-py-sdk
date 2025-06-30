from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class Product(BaseModel):
    """Represents a product with basic details."""

    id: str = Field(..., description="Unique identifier for the product")
    name: str = Field(..., description="Name of the product")
    description: Optional[str] = Field(None, description="Description of the product")
    price: Optional[float] = Field(None, description="Price of the product")
    image_url: Optional[str] = Field(None, description="URL to the product image")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional product metadata"
    )


class Feature(BaseModel):
    """Represents a feature with a category, type, and value."""

    category: str = Field(..., description="The category of the feature.")
    type: str = Field(..., description="The type of the feature.")
    value: str = Field(..., description="The value of the feature.")


class ProductRecommendation(Product):
    """Extends the Product model to include a justification for the recommendation."""

    justification: Optional[str] = Field(
        None, description="The justification for recommending this product."
    )


class HydratedAgentResponse(BaseModel):
    """Represents a structured response from the agent, which can include text,
    product recommendations, and follow-up questions."""

    response_type: str = Field(
        ..., description="The type of response, e.g., 'question', 'recommendation'."
    )
    preamble: Optional[str] = Field(
        None, description="The introductory text of the response."
    )
    follow_up_question: Optional[str] = Field(
        None, description="A follow-up question to the user."
    )
    product_recommendations: List[ProductRecommendation] = Field(
        default_factory=list,
        description="A list of product recommendations.",
    )
    error: Optional[str] = Field(None, description="Any error message that occurred.")


class ChatMessage(BaseModel):
    """Represents a single message in a chat, with a timestamp, role, and content."""

    timestamp: datetime = Field(..., description="The timestamp of the message.")
    role: Literal["user", "assistant"] = Field(
        ..., description="The role of the message sender."
    )
    # Allow content to be either a simple string or any Pydantic BaseModel. This
    # makes the schema flexible for agents that define their own structured
    # response models outside of this base package.
    content: Union[str, Dict[str, Any], BaseModel] = Field(
        ...,
        description="The content of the message. Can be a plain string, a dict, or any Pydantic model.",
    )


class Thread(BaseModel):
    """Represents a conversation thread, with an ID and timestamps for creation and updates."""

    thread_id: str = Field(..., description="The unique identifier for the thread.")
    created_at: datetime = Field(
        ..., description="The timestamp when the thread was created."
    )
    updated_at: datetime = Field(
        ..., description="The timestamp when the thread was last updated."
    )
    title: str = Field(..., description="The title of the thread.")


class ChatHistory(BaseModel):
    """Represents the full history of a chat, including all messages and thread metadata."""

    messages: List[ChatMessage] = Field(
        ..., description="The list of messages in the chat."
    )
    thread_id: str = Field(..., description="The unique identifier for the thread.")
    created_at: datetime = Field(
        ..., description="The timestamp when the thread was created."
    )
    title: str = Field(..., description="The title of the thread.")
