import matplotlib.pyplot as plt

import optimizer_evaluation
from optimizer_evaluation.data_evaluator import Data_Evaluator


import pandas as pd


def _save_nn(nn, name, optimizer):
    nn.save_weights("res/" + name + optimizer + ".h5")


def _plot_eval(name, optimizer, predict, y_test):
    fig, ax = plt.subplots(1, 1)
    ax.plot(predict, color='green', marker=' ', linestyle='dotted')
    ax.plot(y_test.values, color='blue',
            marker='o', linestyle='dotted')
    ax.set_xlabel("Index")
    ax.set_ylabel("Target")
    ax.set_title("Predicted (green) vs. Real Values (blue)")
    plt.savefig("docs/" + name + "/" + name + "_" + optimizer + ".png")


class OptimizerEvaluator(Data_Evaluator):
    optimizer = {'adam': 'adam', 'sgd': 'sgd', 'adagrad': 'adagrad'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate_all(self, metrics_regression=['mae'], metrics_classification=['accuracy'], test=False):

        for name in self.datasets:
            score_list = []
            X_train, X_test, y_train, y_test = self.datasets.get_train_test(
                name)
            for opt in OptimizerEvaluator.optimizer.values():

                nn = self.neural_networks.get(name)

                if self.datasets.get(name).type == 'regression':
                    loss = 'mse'
                    metrics = metrics_regression
                else:
                    loss = 'binary_crossentropy'
                    metrics = metrics_classification

                nn.compile(loss=loss,
                           optimizer=opt,
                           metrics=metrics)

                if test is True:
                    nn.load_weights("res/" + name + opt + ".h5")
                else:
                    nn.fit(X_train.values, y_train.values, epochs=20, batch_size=1)
                    _save_nn(nn, name, opt)

                scores = nn.evaluate(X_test, y_test, verbose=0)
                score_list.append(scores)
                predict = nn.predict(X_test.values, batch_size=1)

                _plot_eval(name, opt, predict, y_test)

            score_df = pd.DataFrame(score_list, columns=nn.metrics_names,
                                    index=list(OptimizerEvaluator.optimizer.values()))
            score_df.to_csv("docs/" + name + "/" + name + "_score.csv")
