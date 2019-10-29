import optimizer_evaluation
from optimizer_evaluation.adam_evaluator import ADAMEvaluator


def test_evaluate_all():
    at = ADAMEvaluator()
    at.evaluate_all()
