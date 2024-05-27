from pydantic import BaseModel, Field
from typing import Optional, Literal


class Example(BaseModel):
    type: Literal["positive", "negative"]
    value: str


class Policy(BaseModel):
    policyType: Literal["text"] = "text"
    value: str
    examples: list[Example] | None = Field(None, max_items=6)
