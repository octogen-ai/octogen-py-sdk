from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from octogen.api.types.search_tool_output import Product


class BaseProductRecommendation(BaseModel):
    """A recommendation from the stylist agent."""

    uuid: str = Field(description="The uuid of the product that is being recommended.")


class BaseComparisonDataItem(BaseModel):
    """Individual item in a comparison dataset."""

    category: str = Field(description="Category of the comparison data")


class ComparisonDataCategory(BaseComparisonDataItem):
    """Individual item in a comparison dataset."""

    category: str = Field(description="Category of the comparison data")
    items: List[BaseProductRecommendation] = Field(
        description="Items to compare. Can be either full product objects or simple UUID references that will be hydrated later."
    )


class HydratedComparisonDataCategory(BaseComparisonDataItem):
    """Individual item in a comparison dataset."""

    category: str = Field(description="Category of the comparison data")
    items: List[Product] = Field(
        description="Items to compare. Can be either full product objects or simple UUID references that will be hydrated later."
    )


class DisplayConfig(BaseModel):
    """Base class for display configurations."""

    display_type: str = Field(description="Type of display configuration")


class TableDisplay(DisplayConfig):
    """Table display configuration."""

    display_type: Literal["table"] = Field(
        description="Type of display configuration", default="table"
    )
    highlighted_features: List[str] = Field(
        description="These MUST be the EXACT name of the feature in the product group that you want to highlight in the comparison. Best for comparing multiple products across standardized features. Use when you have several products that can be compared by specific features. Only include features that have data for most or all products in the table. NEVER include 'image' or 'images' fields - these are handled automatically by the UI.",
        default_factory=lambda: [],
    )
    sort_by: str = Field(
        description="Feature to sort the table by",
        default="price",
    )
    sort_direction: Literal["asc", "desc"] = Field(
        description="Direction to sort the table",
        default="asc",
    )


class ScatterPlotDisplay(DisplayConfig):
    """Scatter plot display configuration."""

    display_type: Literal["scatter"] = Field(
        description="Type of display configuration", default="scatter"
    )
    title: Optional[str] = Field(
        description="Title to display at the top of the scatter plot visualization. Should be concise and descriptive of what the plot is showing.",
        default=None,
    )
    tooltip_features: List[str] = Field(
        description="Features to display in the tooltip, showcase information the user would want to know about the product. ALWAYS include the x_axis and y_axis features in this list - this is REQUIRED. The first two elements of this list MUST be the x_axis and y_axis features. Only include features that contain actual data. NEVER include 'image' or 'images' fields.",
        default_factory=lambda: [],
    )
    x_axis: str = Field(
        description="Feature to display on the x-axis. Best for visualizing relationship between two numeric features. Must contain valid numeric data in all products.",
        default="price",
    )
    y_axis: str = Field(
        description="Feature to display on the y-axis. Best for visualizing relationship between two numeric features. Must contain valid numeric data in all products.",
        default="rating",
    )


class IndividualComparisonDisplay(DisplayConfig):
    """Individual comparison display configuration."""

    display_type: Literal["individual"] = Field(
        description="Type of display configuration", default="individual"
    )
    included_features: List[str] = Field(
        description="Features to include in the comparison. Best for detailed side-by-side comparison of EXACTLY 2 specific products. Only include features that have actual data in BOTH products being compared. Include as many features as possible - include ALL features that have valid data in both products to provide the most comprehensive comparison. This should be an exhaustive list of every useful field with data. NEVER include 'image' or 'images' fields - these are handled automatically by the UI.",
        default_factory=lambda: [],
    )
    highlighted_features: List[str] = Field(
        description="Features to highlight in the comparison. These are the MOST IMPORTANT features to emphasize. Only include features that have data in BOTH products and that you find especially useful to the user. These should be a subset of included_features that deserve special attention. NEVER include 'image' or 'images' fields.",
        default_factory=lambda: [],
    )


class BaseComparisonResponse(BaseModel):
    """Structure for comparison agent responses with data for product comparisons."""

    response_type: Literal["comparison", "freeform_question"] = Field(
        description="The type of response. 'comparison' for structured product comparison, 'freeform_question' for text responses. Use 'comparison' whenever you make a tool call to search_data_for_comparison, and 'freeform_question' for error messages or clarification requests.",
        default="comparison",
    )
    preamble: str = Field(
        description="Introduction or context for the comparison. Should be clear and helpful.",
        default="",
    )
    suggested_display: Optional[
        TableDisplay | ScatterPlotDisplay | IndividualComparisonDisplay
    ] = Field(
        description="Suggested visualization format. Choose exactly ONE display type that best fits the data. For TableDisplay: use when comparing multiple products across standardized features. For ScatterPlotDisplay: use when visualizing relationship between two numeric features. For IndividualComparisonDisplay: use when comparing exactly 2 specific products in detail.",
        default=None,
    )
    analysis: str = Field(
        description="Detailed analysis of the comparison results, highlighting key insights, patterns, and recommendations. Identify the top product choices and why they stand out, explain important trade-offs, highlight surprising findings, and provide a balanced overview in 2-4 sentences.",
        default="",
    )
    suggested_follow_up_questions: List[str] = Field(
        description="A focused list of follow-up questions for the USER to ask YOU that are strictly limited to: 1) Viewing the current data in a different display type (e.g., 'Can I see these products in a table instead?'), 2) Comparing specific products from the current display (for individual comparisons, always name exactly two real products from the display), or 3) Adding other specific categories to the current comparison. Always refer to specific products, features, or categories by name. Do NOT suggest general or creative questions outside these three types. Keep questions concise and directly actionable with the current display data.",
        default=[],
    )


class ComparisonResponse(BaseComparisonResponse):
    comparison_data: List[ComparisonDataCategory] = Field(
        description="Stru ctured data for the comparison. Each search you perform should be organized as its own separate category. Never mix products from different searches into the same category.",
        default_factory=lambda: [],
    )


class HydratedComparisonResponse(BaseComparisonResponse):
    comparison_data: List[HydratedComparisonDataCategory] = Field(
        description="Structured data for the comparison. Each search you perform should be organized as its own separate category. Never mix products from different searches into the same category.",
        default_factory=lambda: [],
    )
