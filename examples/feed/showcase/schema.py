from typing import Literal, Optional

from pydantic import BaseModel, Field

from octogen.api.types.catalog_text_search_params import CatalogTextSearchParams

RESPONSE_TYPE_DESCRIPTION = "The type of response. If the type is freeform_question, the response should only contain the preamble. If the type is search_query, the response should contain the preamble and search query."
PREAMBLE_DESCRIPTION = "The preamble to the response. It should always be present and should be a question if the response_type is freeform_question."
SEARCH_QUERY_DESCRIPTION = "The search query built based on the user's request. Always include when response_type is search_query"


class SearchQuery(BaseModel):
    """Search query that follows the structure of CatalogTextSearchParams with an added explanation field."""

    catalog_params: CatalogTextSearchParams = Field(
        description="The catalog search parameters"
    )
    explanation: str = Field(
        description="Explanation of how the search query was constructed from the user's request"
    )


ERROR_DESCRIPTION = "Error message if the response type is error"


class AgentResponse(BaseModel):
    response_type: Literal["search_query", "freeform_question", "error"] = Field(
        description=RESPONSE_TYPE_DESCRIPTION,
    )
    preamble: Optional[str] = Field(description=PREAMBLE_DESCRIPTION)
    search_query: Optional[SearchQuery] = Field(
        description=SEARCH_QUERY_DESCRIPTION,
        default=None,
    )
    error: Optional[str] = Field(
        description=ERROR_DESCRIPTION,
        default=None,
    )
