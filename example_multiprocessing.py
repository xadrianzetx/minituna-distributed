import random
import time

import minituna_multiprocess as minituna


def objective(trial: minituna.Trial) -> float:
    x = trial.suggest_uniform("x", 0, 10)
    y = trial.suggest_uniform("y", 0, 10)
    time.sleep(random.randint(0, 2))
    return (x - 3) ** 2 + (y - 5) ** 2


if __name__ == "__main__":
    start = time.time()
    study = minituna.create_study()
    study.optimize(objective, 10)
    best_trial = study.best_trial
    elapsed = time.time() - start
    print(
        f"Best trial: value={best_trial.value} params={best_trial.params}. Elapsed: {elapsed:.2f}"
    )
