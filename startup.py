import sys
sys.path.append("$PWD")

import optimizer_evaluation
from optimizer_evaluation.adam_evaluator import ADAMEvaluator

ad_ev = ADAMEvaluator()
ad_ev.evaluate_all()