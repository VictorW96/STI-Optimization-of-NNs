import optimizer_evaluation
from optimizer_evaluation.optimizer_evaluator import OptimizerEvaluator


def test_evaluate_all():
    at = OptimizerEvaluator()
    at.evaluate_all()
