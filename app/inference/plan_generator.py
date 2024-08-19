from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass
from enum import Enum

class ModelClass(Enum):
    SMALL='small'
    LARGE='large'

@dataclass
class GeneratedPlan:
    plan: str
    model_class: ModelClass
    confidence: Optional[float] = None

class PlanGenerator(ABC):

    @abstractmethod
    def generate_plan(self, utterance: str) -> GeneratedPlan:
        pass
