from pydantic import BaseModel, Field, ValidationError

class Citation(BaseModel):
    source_url: str
    relevance_score: float = Field(gt=0.0, le=1.0, description="score between 0 and 1")

class AgentAnalysis(BaseModel):
    agent_name: str
    summary: str = Field(min_length=10)
    citations: list[Citation]
    requires_followup: bool = Field(False)

raw_data = {
    "agent_name": "gemma",
    "summary": "chats with user",
    "citations": [
        {
        "source_url": "https://gemini.google.com/app/079574522e40f469",
        "relevance_score": 0.5
        }
    ],
    "requires_followup": True
}    


def parse_llm_output(raw_json: dict):
    try:
        validated_response = AgentAnalysis(**raw_json)
        print(f"Agent name is {validated_response.agent_name}")
        print(f"source url {validated_response.citations[0].source_url}")

    except ValidationError as e:
        print("validation failed")
        print(e)    


parse_llm_output(raw_data)