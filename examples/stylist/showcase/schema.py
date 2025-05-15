from typing import (
    Dict,
    List,
    Literal,
    Optional,
    TypeAlias,
)

from pydantic import BaseModel, Field

from octogen.api.types.search_tool_output import Product


class StylistProductRecommendation(BaseModel):
    """A recommendation from the stylist agent."""

    uuid: str = Field(description="The uuid of the product that is being recommended.")


Outfit: TypeAlias = Dict[str, StylistProductRecommendation | None]
HydratedOutfit: TypeAlias = Dict[str, Product | None]


class BaseStylistRecommendation(BaseModel):
    """A recommendation from the stylist agent."""

    group_summary: str = Field(
        description="A summary of the group of products that are being recommended. A group could refer to a product type, style, or unique combination of features. Also include justification for why these products are being recommended."
    )


class StylistRecommendation(BaseStylistRecommendation):
    """A recommendation from the stylist agent."""

    products: List[Product] = Field(
        description="The list of products that are being recommended. These are the products from the search tool that meet the criteria for the group summary."
    )


class HydratedStylistRecommendation(BaseStylistRecommendation):
    """A recommendation from the stylist agent."""

    products: List[Product] = Field(
        description="The list of products that are being recommended. These are the products from the search tool that meet the criteria for the group summary."
    )


class BaseStylistAgentResponse(BaseModel):
    """The response from the stylist agent."""

    response_type: Literal["recommendation", "freeform"] = Field(
        description="The type of response. If the type is freeform, the response should only contain the preamble and not the product recommendations."
    )
    preamble: str = Field(
        description="The preamble to the response. It should always be present and should be a question if the response_type is freeform",
    )


class StylistAgentResponse(BaseStylistAgentResponse):
    """The response from the stylist agent."""

    product_recommendations: Optional[List[StylistRecommendation]] = Field(
        description="The list of product recommendations. Always include when response_type is recommendation",
        default=None,
    )
    current_outfit: Optional[Outfit] = Field(
        description="A dictionary of the current outfit. The key is the product type and the value is the pick for that component of the outfit. If we don't have a pick for a component, the value should be None.",
        default=None,
    )


class HydratedStylistAgentResponse(BaseStylistAgentResponse):
    """The response from the stylist agent."""

    product_recommendations: Optional[List[HydratedStylistRecommendation]] = Field(
        description="The list of product recommendations. Always include when response_type is recommendation",
        default=None,
    )
    current_outfit: Optional[HydratedOutfit] = Field(
        description="A dictionary of the current outfit. The key is the product type and the value is the pick for that component of the outfit. If we don't have a pick for a component, the value should be None.",
        default=None,
    )
