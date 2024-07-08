from pydantic import BaseModel, Field
from typing import Optional, Literal


class Example(BaseModel):
    type: Literal["positive", "negative"]
    value: str


class Policy(BaseModel):
    id: str|int
    name: str
    policyType: Literal["text"] = "text"
    value: str
    examples: list[Example] | None = Field(None, max_length=6)
