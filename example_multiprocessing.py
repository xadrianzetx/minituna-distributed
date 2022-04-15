import random
import time

import minituna_multiprocess_pool as minituna


def objective(trial: minituna.Trial) -> float:
    x = trial.suggest_float("x", 0, 10)
    y = trial.suggest_float("y", 0, 10)
    time.sleep(random.randint(0, 2))
    return (x - 3) ** 2 + (y - 5) ** 2


if __name__ == "__main__":
    start = time.time()
    study = minituna.create_study()
    study.optimize(objective, 10, n_jobs=-1)
    best_trial = study.best_trial
    elapsed = time.time() - start
    print(
        f"Best trial: value={best_trial.value} params={best_trial.params}. Elapsed: {elapsed:.2f}"
    )
