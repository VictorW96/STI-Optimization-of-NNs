from optimizer_evaluation import adam_tester

def test_test():
    at = adam_tester.ADAMTester()
    at.test_all()