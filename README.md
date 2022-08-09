# Minituna-distributed

A distributed optimization POC for [Optuna](https://github.com/optuna/optuna), based on fork of awesome [Minituna](https://github.com/CyberAgentAILab/minituna) toy hyperparameter optimization framework.

## Core idea

Much like in [`minituna-multiprocess`](https://github.com/xadrianzetx/minituna-multiprocess), the idea is to distribute execution of trials across workers (in this case different physical machines) as tasks and keep study and its resources (storage, sampler etc.) local in client process (see rationale behind that in `minituna-multiprocess` README file). Task scheduler, client, cluster deployment and coordination primitives (pubsub) are provided by [dask](https://docs.dask.org/en/stable/).

## Implementation

Working example was build on top of `minituna_v3.py` and can be found in `minituna_distributed.py`. Changes include:

* New implementation of `Study.optimize`, which now is responsible for spawning tasks, initializing and maintaining communication with workers
* `DistributedTrial` class which holds communication with main process on the worker side
* Simple communication protocol done via implementations of `Command` base class
* Minimalistic optimization process controller

All examples can be executed out-of-the-box thanks to [local cluster](https://docs.dask.org/en/latest/deploying-python.html#localcluster) provided by dask. However, these are much cooler when running on multiple physical machines. To achieve that, deploy a small [dask cluster](https://docs.dask.org/en/stable/deploying.html) (Raspberry Pis are fine!), instantiate [dask client](https://docs.dask.org/en/latest/futures.html#distributed.Client) and point it at your cluster scheduler. Environment on cluster must more-or-less match your local, including `minituna-distributed` module, so standard `setup.py` script is available to package required files.
