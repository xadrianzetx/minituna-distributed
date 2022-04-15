# Minituna-multiprocess

This is yet another POC for process based parallelism in [Optuna](https://github.com/optuna/optuna), based on fork of awesome [Minituna](https://github.com/CyberAgentAILab/minituna) toy hyperparameter optimization framework.

## Core idea

The idea is to keep study resources (storage, sampler etc.) only in main process and access them sequentially for simplicity, but keep "heavy" parts such as objective function evaluation distributed for efficiecy. This is done by custom `Trial` implemetation, which runs with user defined objective function in worker process and is able to issue simple commands to main process (via pipes) such as suggest parameter or report intermediate value, and recieve responses. Main process is then tasked with queueing and evaluating those commands in sequential manner against study resources. From study perspective we are performing simple sequential optimization, but in reality workload is distributed.

Main characteristics:

* No shared memory between processes
* No need for locks, no race conditions in storage
* Users do not need to invoke special storage instance to enable multiprocessing. All IPC related classes are internal
* 'n_jobs' argument can control number of spawned processes, or can be removed and workers pool controlled automatically
* No changes to existing Optuna public APIs
* Minimal dependencies sent over to workers (user defined objective function + some thin wrapper)
* Simplified (sequential) interaction with callbacks, progress bar etc. in multiprocess mode

Possible drawbacks:

* IPC protocol to maintain (added code complexity)
* Additional safety measures needed to ensure processes are disposed of correctly
* This won't scale to multi machine setups

## Implementation

Working example was build on top of `minituna_v3.py` and can be found in `minituna_multiprocess.py`. Changes include:

* New implementation of `Study.optimize`, which now is responsible for spawning processes, initializing and maintaining communication with workers
* `IPCTrial` class which holds communication with main process on the worker side
* Simple IPC protocol done via implementations of `Command` base class

Alternatively, there is `minituna_multiprocess_pool.py` which also includes `n_jobs` argument in `Study.optimize` and implements simple process pool.

You can run `example_multiprocessing.py` or `example_pruning.py` to see multiprocessing in action.
