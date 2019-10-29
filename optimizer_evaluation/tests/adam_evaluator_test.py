import optimizer_evaluation
from optimizer_evaluation.adam_evaluator import ADAMEvaluator

def evaluate_all_test():
    at = ADAMEvaluator()
    at.evaluate_all()