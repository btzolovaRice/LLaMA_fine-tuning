import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from anthropic import AnthropicBedrock
from .routers.inference import InferenceRouter
from .inference import ClaudePlanGenerator, OllamaPlanGenerator, DynamicPlanGenerator

logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Dependency injection 
# Later could move to a framework, makes testing easier and code cleaner
client = AnthropicBedrock(
    aws_region="us-east-1",
)

ollama_generator = OllamaPlanGenerator(model_name='continua') # can replace with whatever you name yours, or use 'llama3.1'
claude_generator = ClaudePlanGenerator(bedrock_client=client)
dynamic_generator = DynamicPlanGenerator(small_plan_generator=ollama_generator, large_plan_generator=claude_generator)

inference_router = InferenceRouter(dynamic_plan_generator=dynamic_generator)

app.include_router(inference_router)

@app.get("/")
async def root():
  return {"message": "healthy"}
