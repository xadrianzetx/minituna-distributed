import abc
import copy
import ctypes
import math
import pickle
import random
import socket
import threading
import time
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union
import uuid

from dask.distributed import Client
from dask.distributed import Future
from dask.distributed import LocalCluster
from dask.distributed import Queue
from dask.distributed import Variable
import numpy as np


TrialStateType = Literal["running", "completed", "pruned", "failed"]
CategoricalChoiceType = Union[None, bool, int, float, str]


class TrialPruned(Exception):
    ...


class WorkerInterrupted(Exception):
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
    ) -> float:
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


class PickledQueue:
    def __init__(self, name: str) -> None:
        self.q = Queue(name)

    def put(self, value: Any, timeout: Optional[Any] = None) -> None:
        self.q.put(pickle.dumps(value), timeout)

    def get(self, timeout: Optional[Any] = None) -> Any:
        return pickle.loads(self.q.get(timeout))


class OptimizationHeartbeat:
    def __init__(self, manager: "OptimizationManager") -> None:

        # A hook for periodic checkups would be nice as well.
        self._manager = manager

    def ensure_safe_exit(self, future: Future) -> None:
        if future.status in ["error", "cancelled"]:
            # In case future failed before execution of
            # trial wrapper started, we want to avoid
            # manager waiting for its completion forever.
            self._manager.register_trial_exit()
            PickledQueue(self._manager.common_topic).put(EmptyCommand())


class OptimizationManager:
    def __init__(self, n_trials: int) -> None:
        self._n_trials = n_trials
        self._finished_trials = 0
        self.heartbeat = OptimizationHeartbeat(self)
        self.common_topic = str(uuid.uuid4())
        self._private_topics: Dict[int, str] = {}
        self.stop_condition = Variable("stop-condition")
        self.stop_condition.set(False)

    def assign_private_topic(self, trial_id: int) -> str:
        topic = str(uuid.uuid4())
        self._private_topics[trial_id] = topic
        return topic

    def get_private_topic(self, trial_id: int) -> str:
        return self._private_topics[trial_id]

    def register_trial_exit(self) -> None:
        self._finished_trials += 1

    def should_end_optimization(self) -> bool:
        return self._finished_trials == self._n_trials

    def stop_optimization(self) -> None:
        self.stop_condition.set(True)


class BaseCommand(abc.ABC):
    @abc.abstractmethod
    def execute(self, study: "Study", manager: OptimizationManager) -> None:
        ...


class EmptyCommand(BaseCommand):
    def execute(self, study: "Study", manager: OptimizationManager) -> None:
        ...


class SuggestCommand(BaseCommand):
    def __init__(self, trial_id: int, name: str, distribution: BaseDistribution) -> None:
        self.trial_id = trial_id
        self.name = name
        self.distribution = distribution

    def execute(self, study: "Study", manager: OptimizationManager) -> None:
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

        publisher = PickledQueue(manager.get_private_topic(self.trial_id))
        publisher.put(param_value)


class ReportCommand(BaseCommand):
    def __init__(self, trial_id: int, step: int, value: float) -> None:
        self.trial_id = trial_id
        self.step = step
        self.value = value

    def execute(self, study: "Study", manager: OptimizationManager) -> None:
        trial = Trial(study, self.trial_id)
        trial.report(self.value, self.step)


class ShouldPruneCommand(BaseCommand):
    def __init__(self, trial_id: int) -> None:
        self.trial_id = trial_id

    def execute(self, study: "Study", manager: OptimizationManager) -> None:
        trial = Trial(study, self.trial_id)
        publisher = PickledQueue(manager.get_private_topic(self.trial_id))
        publisher.put(trial.should_prune())


class TrialFinishedCommand(BaseCommand):
    def __init__(self, trial_id: int, value: float, host: str) -> None:
        self.trial_id = trial_id
        self.value = value
        self.host = host  # Just for show :^)

    def execute(self, study: "Study", manager: OptimizationManager) -> None:
        study.storage.set_trial_value(self.trial_id, self.value)
        study.storage.set_trial_state(self.trial_id, "completed")
        manager.register_trial_exit()
        print(f"trial_id={self.trial_id} is completed with value={self.value} by {self.host}")


class TrialFailedCommand(BaseCommand):
    def __init__(self, trial_id: int, e: Exception) -> None:
        self.trial_id = trial_id
        self.e = e

    def execute(self, study: "Study", manager: OptimizationManager) -> None:
        study.storage.set_trial_state(self.trial_id, "failed")
        manager.register_trial_exit()
        print(f"trial_id={self.trial_id} is failed by {self.e}")


class TrialPrunedCommand(BaseCommand):
    def __init__(self, trial_id: int) -> None:
        self.trial_id = trial_id

    def execute(self, study: "Study", manager: OptimizationManager) -> None:
        frozen_trial = study.storage.get_trial(self.trial_id)
        last_step = frozen_trial.last_step
        assert last_step is not None
        value = frozen_trial.intermediate_values[last_step]

        study.storage.set_trial_value(self.trial_id, value)
        study.storage.set_trial_state(self.trial_id, "pruned")
        manager.register_trial_exit()
        print(f"trial_id={self.trial_id} is pruned at step={last_step} value={value}")


class DistributedTrial(Trial):
    def __init__(self, trial_id: int, common_topic: str, private_topic: str) -> None:
        self.trial_id = trial_id
        self.common_topic = common_topic
        self.private_topic = private_topic
        self._publisher: Optional[PickledQueue] = None
        self._subscriber: Optional[PickledQueue] = None

    @property
    def publisher(self) -> Queue:
        # We need to hold reference to publisher/subscriber
        # for the entire lifetime of a task, otherwise topic
        # could get garbage collected.
        if self._publisher is None:
            self._publisher = PickledQueue(self.common_topic)
        return self._publisher

    @property
    def subscriber(self) -> PickledQueue:
        if self._subscriber is None:
            self._subscriber = PickledQueue(self.private_topic)
        return self._subscriber

    def _suggest(self, name: str, distribution: BaseDistribution) -> Any:
        cmd = SuggestCommand(self.trial_id, name, distribution)
        self.publisher.put(cmd)
        param_value = self.subscriber.get()
        return param_value

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: Optional[float] = None,
        log: bool = False,
    ) -> float:
        return self._suggest(name, FloatDistribution(low, high, log, step))

    def suggest_categorical(
        self, name: str, choices: List[CategoricalChoiceType]
    ) -> CategoricalChoiceType:
        return self._suggest(name, CategoricalDistribution(choices=choices))

    def report(self, value: float, step: int) -> None:
        cmd = ReportCommand(self.trial_id, step, value)
        self.publisher.put(cmd)

    def should_prune(self) -> bool:
        cmd = ShouldPruneCommand(self.trial_id)
        self.publisher.put(cmd)
        should_prune = self.subscriber.get()
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
                return self.rng.randint(int(distribution.low), int(distribution.high))
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
    def __init__(self, storage: Storage, sampler: Sampler, pruner: Pruner, client: Client) -> None:
        self.storage = storage
        self.sampler = sampler
        self.pruner = pruner
        self.client = client

    def optimize(self, objective: Callable[[DistributedTrial], float], n_trials: int) -> None:
        manager = OptimizationManager(n_trials)
        commands = PickledQueue(manager.common_topic)
        trial_ids = [self.storage.create_new_trial() for _ in range(n_trials)]
        trials = [
            DistributedTrial(id, manager.common_topic, manager.assign_private_topic(id))
            for id in trial_ids
        ]

        func = _distributable(
            objective, with_supervisor=not isinstance(self.client.cluster, LocalCluster)
        )

        # We need to hold reference to futures, even though we technically don't need them,
        # otherwise task associated with them will be killed by scheduler. We can't just
        # fire and forget, sice we want to avoid orphaned trials if main process goes down.
        futures = self.client.map(func, trials)
        for future in futures:
            future.add_done_callback(manager.heartbeat.ensure_safe_exit)

        try:
            self._event_loop(manager, commands)

        except KeyboardInterrupt:
            # Local cluster is destroyed on process exit anyway so we don't need to interrupt.
            if not isinstance(self.client.cluster, LocalCluster):
                self._handle_interrupt(manager, futures)
            raise

    def _event_loop(self, manager: OptimizationManager, commands: PickledQueue) -> None:
        while True:
            command = commands.get()
            command.execute(self, manager)
            if manager.should_end_optimization():
                break

    def _handle_interrupt(self, manager: OptimizationManager, futures: List[Future]) -> None:
        for future in futures:
            future.cancel()

        # FIXME: Dask variable needs to stay in scope until all
        # workers read its final state, otherwise supervisor attempts to
        # read a flag that no longer exists. Obviously, doing it by sleeping
        # and hoping all workers had a chance to exit is a nasty nasty hack. :^)
        manager.stop_optimization()
        time.sleep(1.0)

    @property
    def best_trial(self) -> Optional[FrozenTrial]:
        return self.storage.get_best_trial()


def _distributable(
    func: Callable[[DistributedTrial], float], with_supervisor: bool
) -> Callable[[DistributedTrial], None]:
    def _objective_wrapper(trial: DistributedTrial) -> None:
        cmd: BaseCommand
        stop_flag = threading.Event()
        if with_supervisor:
            tid = threading.get_ident()
            threading.Thread(target=_supervisor, args=(tid, stop_flag), daemon=True).start()

        try:
            value_or_values = func(trial)
            host = socket.gethostname()
            cmd = TrialFinishedCommand(trial.trial_id, value_or_values, host)
            trial.publisher.put(cmd)

        except TrialPruned:
            cmd = TrialPrunedCommand(trial.trial_id)
            trial.publisher.put(cmd)

        except WorkerInterrupted:
            print(f"Trial {trial.trial_id} interrupted by supervisor.")

        except Exception as e:
            cmd = TrialFailedCommand(trial.trial_id, e)
            trial.publisher.put(cmd)

        finally:
            stop_flag.set()

    return _objective_wrapper


def _supervisor(thread_id: int, parent_exit: threading.Event) -> None:
    stop_condition = Variable("stop-condition")
    while True:
        time.sleep(0.1)
        if parent_exit.is_set():
            break

        if stop_condition.get():
            # https://gist.github.com/liuw/2407154
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread_id), ctypes.py_object(WorkerInterrupted)
            )
            break


def create_study(
    storage: Optional[Storage] = None,
    sampler: Optional[Sampler] = None,
    pruner: Optional[Pruner] = None,
    client: Optional[Client] = None,
) -> Study:
    return Study(
        storage=storage or Storage(),
        sampler=sampler or Sampler(),
        pruner=pruner or Pruner(),
        client=client or Client(),
    )
