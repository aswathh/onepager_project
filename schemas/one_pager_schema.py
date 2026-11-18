
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class CompanyInformation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_name: Optional[str] = Field(
        default=None,
        description="Official name of the company."
    )

    overview: Optional[str] = Field(
        default=None,
        description="Brief summary of the company's business, core technology, products, and purpose."
    )

    headquarters: Optional[str] = Field(
        default=None,
        description="Company's headquarters location, including city and country when available."
    )

    employee_count: Optional[str] = Field(
        default=None,
        description="Number of employees or an explicitly stated employee range."
    )

    funding: Optional[str] = Field(
        default=None,
        description="Funding raised by the company, including amount, currency, and funding round when stated. Do not confuse funding with revenue."
    )

    revenue: Optional[str] = Field(
        default=None,
        description="Company revenue, including amount, currency, and reporting period when stated. Do not infer missing currency or period."
    )

    gtm_partners: Optional[str] = Field(
        default=None,
        description="Go-to-market partners involved in selling, distributing, or implementing the company's products."
    )

    key_customers: Optional[str] = Field(
        default=None,
        description="Named organizations that actually use or purchase the company's products or services. Do not list partners as customers unless explicitly stated."
    )

    competitors: Optional[str] = Field(
        default=None,
        description="Companies or products identified as competitors in the same or a closely related market."
    )


class ProductInformation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_name: Optional[str] = Field(
        default=None,
        description="Official name of the product."
    )

    description: Optional[str] = Field(
        default=None,
        description="Brief explanation of what the product does, its main capabilities, and the problem it solves."
    )

    vertical_market: Optional[str] = Field(
        default=None,
        description="Industry or business sector the product serves."
    )

    route_to_market: Optional[str] = Field(
        default=None,
        description="Commercial approach used to sell or distribute the product. This is not the technical deployment environment."
    )

    inference_hardware: Optional[str] = Field(
        default=None,
        description="Hardware used to run model inference. Include specific hardware models only when stated."
    )

    deployment_type: Optional[str] = Field(
        default=None,
        description="Where or how the product is deployed. Do not confuse this with route to market."
    )

    training_hardware: Optional[str] = Field(
        default=None,
        description="Hardware used to train or fine-tune models."
    )

    models: Optional[str] = Field(
        default=None,
        description="AI or machine-learning models used by the product, including model names and architectures when stated."
    )

    ai_framework: Optional[str] = Field(
        default=None,
        description="Frameworks and libraries used to develop, train, or fine-tune AI models."
    )

    serving_framework: Optional[str] = Field(
        default=None,
        description="Frameworks or runtimes used to serve, optimize, or execute models for inference."
    )

    additional_ai_software_stack: Optional[str] = Field(
        default=None,
        description="Other AI and data infrastructure used by the product."
    )


class OnePagerExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: CompanyInformation
    product: ProductInformation


