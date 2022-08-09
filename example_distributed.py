# type: ignore

import random
import time

import minituna_distributed as minituna


def objective(trial):
    x = trial.suggest_float("x", 0, 10)
    y = trial.suggest_float("y", 0, 10)
    time.sleep(random.randint(0, 2))
    return (x - 3) ** 2 + (y - 5) ** 2


if __name__ == "__main__":
    # This example is much better when executed on dask cluster,
    # especially one made up of couple Raspberry Pis :^)
    # https://docs.dask.org/en/stable/deploying.html

    # from dask.distributed import Client
    # client = Client(<your.dask.scheduler.ip>)
    # study = minituna.create_study(client=client)
    start = time.time()
    study = minituna.create_study()
    study.optimize(objective, 10)
    best_trial = study.best_trial
    assert best_trial is not None
    elapsed = time.time() - start
    print(
        f"Best trial: value={best_trial.value} params={best_trial.params}. Elapsed: {elapsed:.2f}"
    )
