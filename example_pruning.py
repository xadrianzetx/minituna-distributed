# type: ignore

from dask.distributed import Client
import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection

import minituna_distributed as minituna


def objective(trial):
    iris = sklearn.datasets.load_iris()
    classes = list(set(iris.target))
    train_x, valid_x, train_y, valid_y = sklearn.model_selection.train_test_split(
        iris.data, iris.target, test_size=0.25
    )

    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    clf = sklearn.linear_model.SGDClassifier(alpha=alpha)

    for step in range(100):
        clf.partial_fit(train_x, train_y, classes=classes)

        # Report intermediate objective value.
        intermediate_value = clf.score(valid_x, valid_y)
        trial.report(intermediate_value, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise minituna.TrialPruned()

    return clf.score(valid_x, valid_y)


if __name__ == "__main__":
    client = Client("localhost:8786")
    study = minituna.create_study(client=client)
    study.optimize(objective, 30)

    best_trial = study.best_trial
    print(f"Best trial: value={best_trial.value} params={best_trial.params}")
