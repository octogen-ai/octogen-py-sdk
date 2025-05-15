from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from octogen.api.types.search_tool_output import Product

RESPONSE_TYPE_DESCRIPTION = "The type of response. If the type is freeform_question, the response should only contain the preamble and not the product recommendations or follow-up question. If the type is recommendation, the response should contain the preamble, product recommendations, and follow-up question."
PREAMBLE_DESCRIPTION = "The preamble to the response. It should always be present and should be a question if the response_type is freeform_question."
PRODUCT_RECOMMENDATIONS_DESCRIPTION = (
    "The product recommendations. Always include when response_type is recommendation"
)
FOLLOW_UP_QUESTION_DESCRIPTION = "Optional follow-up question to refine recommendations. Always include when response_type is recommendation"


class AgentRecommendation(BaseModel):
    uuid: str
    justification: str


class ProductRecommendation(Product):
    justification: str


class AgentResponse(BaseModel):
    response_type: Literal["recommendation", "freeform_question"] = Field(
        description=RESPONSE_TYPE_DESCRIPTION,
    )
    preamble: str = Field(
        description=PREAMBLE_DESCRIPTION,
    )
    product_recommendations: Optional[List[AgentRecommendation]] = Field(
        description=PRODUCT_RECOMMENDATIONS_DESCRIPTION,
        default=None,
    )
    follow_up_question: Optional[str] = Field(
        description=FOLLOW_UP_QUESTION_DESCRIPTION,
        default=None,
    )


ERROR_DESCRIPTION = "Error message if the response type is error"


class HydratedAgentResponse(BaseModel):
    response_type: Literal["recommendation", "freeform_question", "error"] = Field(
        description=RESPONSE_TYPE_DESCRIPTION,
    )
    preamble: Optional[str] = Field(description=PREAMBLE_DESCRIPTION)
    product_recommendations: Optional[List[ProductRecommendation]] = Field(
        description=PRODUCT_RECOMMENDATIONS_DESCRIPTION,
        default=None,
    )
    follow_up_question: Optional[str] = Field(
        description=FOLLOW_UP_QUESTION_DESCRIPTION,
        default=None,
    )
    error: Optional[str] = Field(
        description=ERROR_DESCRIPTION,
        default=None,
    )
