import abc
import copy
import math
from multiprocessing import Pipe
from multiprocessing import Process
from multiprocessing.connection import Connection
from multiprocessing.connection import wait
import random
import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import numpy as np


TrialStateType = Literal["running", "completed", "pruned", "failed"]
CategoricalChoiceType = Union[None, bool, int, float, str]


class TrialPruned(Exception):
    ...


class BaseDistribution(abc.ABC):
    @abc.abstractmethod
    def to_internal_repr(self, external_repr: Any) -> float:
        ...

    @abc.abstractmethod
    def to_external_repr(self, internal_repr: float) -> Any:
        ...


class FloatDistribution(BaseDistribution):
    def __init__(
        self, low: float, high: float, log: bool = False, step: Optional[float] = None
    ) -> None:
        self.low = low
        self.high = high
        self.log = log
        self.step = step

    def to_internal_repr(self, external_repr: Any) -> float:
        if self.step is not None:
            return float(external_repr)
        return external_repr

    def to_external_repr(self, internal_repr: float) -> Any:
        if self.step is not None:
            return int(internal_repr)
        return internal_repr


class CategoricalDistribution(BaseDistribution):
    def __init__(self, choices: List[CategoricalChoiceType]) -> None:
        self.choices = choices

    def to_internal_repr(self, external_repr: Any) -> float:
        return self.choices.index(external_repr)

    def to_external_repr(self, internal_repr: float) -> Any:
        return self.choices[int(internal_repr)]


class FrozenTrial:
    def __init__(self, trial_id: int, state: TrialStateType) -> None:
        self.trial_id = trial_id
        self.state = state
        self.value: Optional[float] = None
        self.intermediate_values: Dict[int, float] = {}
        self.internal_params: Dict[str, float] = {}
        self.distributions: Dict[str, BaseDistribution] = {}

    @property
    def is_finished(self) -> bool:
        return self.state != "running"

    @property
    def params(self) -> Dict[str, Any]:
        external_repr = {}
        for param_name in self.internal_params:
            distribution = self.distributions[param_name]
            internal_repr = self.internal_params[param_name]
            external_repr[param_name] = distribution.to_external_repr(internal_repr)
        return external_repr

    @property
    def last_step(self) -> Optional[int]:
        if len(self.intermediate_values) == 0:
            return None
        else:
            return max(self.intermediate_values.keys())


class Storage:
    def __init__(self) -> None:
        self.trials: List[FrozenTrial] = []

    def create_new_trial(self) -> int:
        trial_id = len(self.trials)
        trial = FrozenTrial(trial_id=trial_id, state="running")
        self.trials.append(trial)
        return trial_id

    def get_all_trials(self) -> List[FrozenTrial]:
        return copy.deepcopy(self.trials)

    def get_trial(self, trial_id: int) -> FrozenTrial:
        return copy.deepcopy(self.trials[trial_id])

    def get_best_trial(self) -> Optional[FrozenTrial]:
        completed_trials = [t for t in self.trials if t.state == "completed"]
        best_trial = min(completed_trials, key=lambda t: cast(float, t.value))
        return copy.deepcopy(best_trial)

    def set_trial_value(self, trial_id: int, value: float) -> None:
        trial = self.trials[trial_id]
        assert not trial.is_finished, "cannot update finished trials"
        trial.value = value

    def set_trial_state(self, trial_id: int, state: TrialStateType) -> None:
        trial = self.trials[trial_id]
        assert not trial.is_finished, "cannot update finished trials"
        trial.state = state

    def set_trial_param(
        self, trial_id: int, name: str, distribution: BaseDistribution, value: float
    ) -> None:
        trial = self.trials[trial_id]
        assert not trial.is_finished, "cannot update finished trials"
        trial.distributions[name] = distribution
        trial.internal_params[name] = value

    def set_trial_intermediate_value(self, trial_id: int, step: int, value: float) -> None:
        trial = self.trials[trial_id]
        assert not trial.is_finished, "cannot update finished trials"
        trial.intermediate_values[step] = value


class Trial:
    def __init__(self, study: "Study", trial_id: int):
        self.study = study
        self.trial_id = trial_id
        self.state = "running"

    def _suggest(self, name: str, distribution: BaseDistribution) -> Any:
        storage = self.study.storage

        trial = storage.get_trial(self.trial_id)
        param_value = self.study.sampler.sample_independent(self.study, trial, name, distribution)
        param_value_in_internal_repr = distribution.to_internal_repr(param_value)
        storage.set_trial_param(self.trial_id, name, distribution, param_value_in_internal_repr)
        return param_value

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: Optional[float] = None,
        log: bool = False,
    ):
        return self._suggest(name, FloatDistribution(low, high, log, step))

    def suggest_categorical(
        self, name: str, choices: List[CategoricalChoiceType]
    ) -> CategoricalChoiceType:
        return self._suggest(name, CategoricalDistribution(choices=choices))

    def report(self, value: float, step: int) -> None:
        self.study.storage.set_trial_intermediate_value(self.trial_id, step, value)

    def should_prune(self) -> bool:
        trial = self.study.storage.get_trial(self.trial_id)
        return self.study.pruner.prune(self.study, trial)


class Command(abc.ABC):
    @abc.abstractmethod
    def execute(self, study: "Study", conn: Connection):
        ...


class Suggest(Command):
    def __init__(self, trial_id: int, name: str, distribution: BaseDistribution) -> None:
        self.trial_id = trial_id
        self.name = name
        self.distribution = distribution

    def execute(self, study: "Study", conn: Connection) -> None:
        trial = Trial(study, self.trial_id)
        if isinstance(self.distribution, FloatDistribution):
            param_value = trial.suggest_float(
                self.name,
                self.distribution.low,
                self.distribution.high,
                step=self.distribution.step,
                log=self.distribution.log,
            )
        elif isinstance(self.distribution, CategoricalDistribution):
            param_value = trial.suggest_categorical(self.name, self.distribution.choices)
        else:
            raise ValueError("Unknown distribution")
        conn.send(param_value)


class Report(Command):
    def __init__(self, trial_id: int, step: int, value: float) -> None:
        self.trial_id = trial_id
        self.step = step
        self.value = value

    def execute(self, study: "Study", conn: Connection) -> None:
        trial = Trial(study, self.trial_id)
        trial.report(self.value, self.step)


class ShouldPrune(Command):
    def __init__(self, trial_id: int) -> None:
        self.trial_id = trial_id

    def execute(self, study: "Study", conn: Connection) -> None:
        trial = Trial(study, self.trial_id)
        conn.send(trial.should_prune())


class Finish(Command):
    def __init__(self, trial_id: int, value: float) -> None:
        self.trial_id = trial_id
        self.value = value

    def execute(self, study: "Study", conn: Connection) -> None:
        study.storage.set_trial_value(self.trial_id, self.value)
        study.storage.set_trial_state(self.trial_id, "completed")
        print(f"trial_id={self.trial_id} is completed with value={self.value}")


class Failed(Command):
    def __init__(self, trial_id: int, e: Exception) -> None:
        self.trial_id = trial_id
        self.e = e

    def execute(self, study: "Study", conn: Connection) -> None:
        study.storage.set_trial_state(self.trial_id, "failed")
        print(f"trial_id={self.trial_id} is failed by {self.e}")


class Pruned(Command):
    def __init__(self, trial_id: int) -> None:
        self.trial_id = trial_id

    def execute(self, study: "Study", conn: Connection) -> None:
        frozen_trial = study.storage.get_trial(self.trial_id)
        last_step = frozen_trial.last_step
        assert last_step is not None
        value = frozen_trial.intermediate_values[last_step]

        study.storage.set_trial_value(self.trial_id, value)
        study.storage.set_trial_state(self.trial_id, "pruned")
        print(f"trial_id={self.trial_id} is pruned at step={last_step} value={value}")


class IPCTrial:
    def __init__(self, trial_id: int, conn: Connection) -> None:
        self.trial_id = trial_id
        self.conn = conn

    def _suggest(self, name: str, distribution: BaseDistribution) -> Any:
        try:
            cmd = Suggest(self.trial_id, name, distribution)
            self.conn.send(cmd)
            param_value = self.conn.recv()
        except EOFError:
            raise

        return param_value

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: Optional[float] = None,
        log: bool = False,
    ):
        return self._suggest(name, FloatDistribution(low, high, log, step))

    def suggest_categorical(
        self, name: str, choices: List[CategoricalChoiceType]
    ) -> CategoricalChoiceType:
        return self._suggest(name, CategoricalDistribution(choices=choices))

    def report(self, value: float, step: int) -> None:
        cmd = Report(self.trial_id, step, value)
        self.conn.send(cmd)

    def should_prune(self) -> bool:
        try:
            cmd = ShouldPrune(self.trial_id)
            self.conn.send(cmd)
            should_prune = self.conn.recv()
        except EOFError:
            raise

        return should_prune


class Sampler:
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)

    def sample_independent(
        self,
        study: "Study",
        trial: FrozenTrial,
        name: str,
        distribution: BaseDistribution,
    ) -> Any:
        if isinstance(distribution, FloatDistribution):
            if distribution.log:
                log_low = math.log(distribution.low)
                log_high = math.log(distribution.high)
                return math.exp(self.rng.uniform(log_low, log_high))
            elif distribution.step is not None:
                return self.rng.randint(distribution.low, distribution.high)
            else:
                return self.rng.uniform(distribution.low, distribution.high)
        elif isinstance(distribution, CategoricalDistribution):
            index = self.rng.randint(0, len(distribution.choices) - 1)
            return distribution.choices[index]
        else:
            raise ValueError("unsupported distribution")


class Pruner:
    def __init__(self, n_startup_trials: int = 5, n_warmup_steps: int = 0) -> None:
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps

    def prune(self, study: "Study", trial: FrozenTrial) -> bool:
        all_trials = study.storage.get_all_trials()
        n_trials = len([t for t in all_trials if t.state == "completed"])

        if n_trials < self.n_startup_trials:
            return False

        last_step = trial.last_step
        if last_step is None or last_step < self.n_warmup_steps:
            return False

        # Median pruning
        others = [
            t.intermediate_values[last_step]
            for t in all_trials
            if last_step in t.intermediate_values
        ]
        median = np.nanmedian(np.array(others))
        return trial.intermediate_values[last_step] > median


class Study:
    def __init__(self, storage: Storage, sampler: Sampler, pruner: Pruner) -> None:
        self.storage = storage
        self.sampler = sampler
        self.pruner = pruner

    def optimize(self, objective: Callable[[Trial], float], n_trials: int) -> None:
        def _objective_wrapper(trial: IPCTrial) -> None:
            try:
                value_or_values = objective(trial)
                cmd = Finish(trial.trial_id, value_or_values)
                trial.conn.send(cmd)

            except TrialPruned:
                cmd = Pruned(trial.trial_id)
                trial.conn.send(cmd)

            except EOFError:
                # Master was killed and we are orphaned now.
                # At least I hope that pipes are closed to raise this
                # exception when it happens.
                sys.exit(0)

            except Exception as e:
                cmd = Failed(trial.trial_id, e)
                trial.conn.send(cmd)

            finally:
                trial.conn.close()

        connections: List[Connection] = []
        for _ in range(n_trials):
            master, worker = Pipe()
            trial_id = self.storage.create_new_trial()
            trial = IPCTrial(trial_id, worker)

            # Alternatively introduce handlers for SIGINT and SIGKILL
            # to make sure all child processes are killed before we crash.
            # https://docs.python.org/3/library/signal.html#note-on-signal-handlers-and-exceptions
            p = Process(target=_objective_wrapper, args=(trial,))
            p.daemon = True

            # Closing our end of worker connection as in
            # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.connection.wait
            connections.append(master)
            p.start()
            worker.close()

        while connections:
            for conn in wait(connections):
                try:
                    # Worker sends command that should be executed
                    # with study resources (storage, sampler etc.) along
                    # with data produced by objective function required to
                    # perform the operation. Connection to the worker is
                    # also included since we might need to send a response.
                    command: Command = conn.recv()
                    command.execute(self, conn)

                except EOFError:
                    # Raised when worker process closes connection and exits.
                    # At the moment only workers control connection closing,
                    # which is probably bad, since master can be interrupted,
                    # and should cleanup before exiting.
                    connections.remove(conn)

    @property
    def best_trial(self) -> Optional[FrozenTrial]:
        return self.storage.get_best_trial()


def create_study(
    storage: Optional[Storage] = None,
    sampler: Optional[Sampler] = None,
    pruner: Optional[Pruner] = None,
) -> Study:
    return Study(
        storage=storage or Storage(),
        sampler=sampler or Sampler(),
        pruner=pruner or Pruner(),
    )
