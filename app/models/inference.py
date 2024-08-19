from pydantic import BaseModel

class GeneratePlanRequest(BaseModel):
    utterance: str

class GeneratePlanResponse(BaseModel):
    utterance: str
    plan: str
    model_class: str