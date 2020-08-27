import minituna


def objective(trial: minituna.Trial) -> float:
    x = trial.suggest_uniform("x", 0, 10)
    y = trial.suggest_uniform("y", 0, 10)
    return (x - 3) ** 2 + (y - 5) ** 2


if __name__ == "__main__":
    study = minituna.create_study()
    study.optimize(objective, 10)
    print("Best trial:", study.best_trial)