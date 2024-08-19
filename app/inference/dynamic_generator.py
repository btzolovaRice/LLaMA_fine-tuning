import numpy as np
from collections import deque
import logging
from .plan_generator import PlanGenerator, GeneratedPlan

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RequestTracker:
    def __init__(self, max_requests: int = 100000):
        self._total_requests = 0
        self._large_model_requests = 0
        self._max_requests = max_requests

    def increment_total(self):
        self._total_requests += 1
        self._check_reset()

    def increment_large_model(self):
        self._large_model_requests += 1
        self._check_reset()

    def get_total_requests(self) -> int:
        return self._total_requests

    def get_large_model_requests(self) -> int:
        return self._large_model_requests

    def get_delegation_ratio(self) -> float:
        if self._total_requests == 0:
            return 0
        return self._large_model_requests / self._total_requests

    def _check_reset(self):
        if self._total_requests >= self._max_requests:
            self._reset_counters()

    def _reset_counters(self):
        logging.info("Resetting counters due to reaching the maximum request limit.")
        self._total_requests = 0
        self._large_model_requests = 0

class DynamicPlanGenerator(PlanGenerator):

    def __init__(self, small_plan_generator: PlanGenerator, large_plan_generator: PlanGenerator, max_delegation_ratio: float = 0.2, max_requests: int = 100000):
        super().__init__()
        self._small_generator = small_plan_generator
        self._large_generator = large_plan_generator
        self._max_delegation_ratio = max_delegation_ratio
        self._tracker = RequestTracker(max_requests)
        self._confidence_scores = deque(maxlen=100)

    def generate_plan(self, utterance: str) -> GeneratedPlan:
        self._tracker.increment_total()

        small_plan: GeneratedPlan = self._small_generator.generate_plan(utterance)

        logging.info(f"small model generated plan with confidence {small_plan.confidence}")
        
        self._confidence_scores.append(small_plan.confidence)
        
        p20_threshold = np.percentile(self._confidence_scores, 20)

        logging.info(f"p20 confidence is {p20_threshold}")

        delegation_ratio = self._tracker.get_delegation_ratio()

        logging.info(f"delegation ratio: {delegation_ratio}")

        if small_plan.confidence <= p20_threshold and delegation_ratio < self._max_delegation_ratio:
            logging.info("delegating to large model")
            large_plan: GeneratedPlan = self._large_generator.generate_plan(utterance)
            self._tracker.increment_large_model()
            return large_plan

        return small_plan
