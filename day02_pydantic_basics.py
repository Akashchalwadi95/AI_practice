from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

# Define a sub-model for LLM execution metrics
class ModelPerformance(BaseModel):
    tokens_used: int = Field(gt=0, description="Total tokens consumed")
    latency_ms: float = Field(gt=0, description="Response time in milliseconds")

# Define the main response schema
class LLMResponseSchema(BaseModel):
    request_id: str
    provider: str
    prompt: str
    output_text: str
    confidence_score: float = Field(ge=0.0, le=1.0, description="Score between 0.0 and 1.0")
    tags: List[str] = Field(default_factory=list)
    performance: ModelPerformance  # Nested Pydantic model
    cached: bool = False

# --- Validating Data ---
raw_data = {
    "request_id": "req-9921",
    "provider": "Gemini",
    "prompt": "Explain Pydantic v2",
    "output_text": "Pydantic is a data validation library...",
    "confidence_score": 0.95,
    "tags": ["ai", "python", "backend"],
    "performance": {
        "tokens_used": 150,
        "latency_ms": 230.5
    }
}

try:
    # Parsing dict into a validated Pydantic object
    validated_response = LLMResponseSchema(**raw_data)
    print(" Validation Successful!")
    print(f"Request ID: {validated_response.request_id}")
    print(f"Tokens Used: {validated_response.performance.tokens_used}")
    
    # Dump back to clean JSON/Dict format
    print("\n Exporting back to JSON Schema:")
    print(validated_response.model_dump_json(indent=2))

except ValidationError as e:
    print(" Validation Failed:")
    print(e)