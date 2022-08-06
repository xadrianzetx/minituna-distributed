# type: ignore

import minituna_multiprocess as minituna


def objective(trial):
    x = trial.suggest_float("x", 0, 10)
    y = trial.suggest_float("y", 0, 10)
    return (x - 3) ** 2 + (y - 5) ** 2


if __name__ == "__main__":
    study = minituna.create_study()
    study.optimize(objective, 10, n_jobs=-1)
    best_trial = study.best_trial
    print(f"Best trial: value={best_trial.value} params={best_trial.params}")
