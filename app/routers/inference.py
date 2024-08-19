from typing import Annotated, List, Optional
from enum import Enum
from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import logging
from ..inference.dynamic_generator import DynamicPlanGenerator
from ..models import GeneratePlanRequest, GeneratePlanResponse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

security = HTTPBearer()

class InferenceRouter(APIRouter):
    def __init__(self, dynamic_plan_generator: DynamicPlanGenerator):
        def auth(creds: Annotated[HTTPAuthorizationCredentials, Depends(security)]):
            # TODO: implement authN/Z mechanism
            # could extract identity in a middleware as well to use userId in business logic
            pass
        
        super().__init__(
                prefix=f'/api/inference', # could version if we want as well
                tags=["inference"],
                # dependencies=[Depends(auth)] # add once ready for authN/Z
        )

        @self.post('/generate-plan')
        def generate_plan(request: GeneratePlanRequest) -> GeneratePlanResponse:
            plan = dynamic_plan_generator.generate_plan(request.utterance)
            return GeneratePlanResponse(utterance=request.utterance, plan=plan.plan, model_class=plan.model_class)


