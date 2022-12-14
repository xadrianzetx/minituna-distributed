# type: ignore

from dask.distributed import Client
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm

import minituna_distributed as minituna


def objective(trial):
    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target

    classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
    else:
        rf_max_depth = trial.suggest_float("rf_max_depth", 2, 32)
        classifier_obj = sklearn.ensemble.RandomForestClassifier(
            max_depth=int(rf_max_depth), n_estimators=10
        )

    score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return 1 - accuracy


if __name__ == "__main__":
    client = Client("localhost:8786")
    study = minituna.create_study(client=client)
    study.optimize(objective, 100)

    best_trial = study.best_trial
    print(f"Best trial: value={best_trial.value} params={best_trial.params}")
