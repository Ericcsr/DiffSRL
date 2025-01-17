from typing import Any, List, Union
import os, subprocess, sys

import torch

from mpi4py import MPI
import numpy as np
import pynvml

pynvml.nvmlInit()
NUM_CUDA = pynvml.nvmlDeviceGetCount()
pynvml.nvmlShutdown()

def best_mpi_subprocess_num(batchSize: int, procPerGPU: int = 1) -> int:
    """ Determine the most suitable number of sub processes

    The method will returns the minimum of the following:
    * batch size;
    * number of GPU cores * procPerGPU and
    * number of CPU cores. 

    :param batch_size: the size of each batch
    :param proPerGPU: how many processes on each GPU
    :return: the minimum of the above three
    """
    cpu_num = os.cpu_count()
    return min(batchSize, cpu_num, NUM_CUDA * procPerGPU)

def _abs_path_2_module_name(absPath:str) -> str:
    absPath = absPath.rstrip('.py').split('/')
    repoRoot = os.path.abspath('.').split('/') # path to where the python -m is originally executed
    i = 0
    while i < min(len(absPath), len(repoRoot)):
        if absPath[i] == repoRoot[i]: i += 1
        else:                         break
    return '.'.join(absPath[i:])
    
def _all_reduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)

def _op(x: Union[torch.Tensor, np.ndarray, List, Any], op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    if isinstance(x, torch.Tensor):
        x = np.asarray(x.cpu(), dtype=np.float32)
    else:
        x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    _all_reduce(x, buff, op=op)
    return buff[0] if scalar else buff

def _sum(x: Union[torch.Tensor, np.ndarray, List, Any]):
    return _op(x, MPI.SUM)

def mpi_fork(n: int, bind_to_core: bool=False) -> None:
    """
    Re-launches the current script with workers linked by MPI.
    Also, terminates the original process that launched it.
    Taken almost without modification from the Baselines function of the
    `same name`_.
    .. _`same name`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py
    Args:
        n (int): Number of process to split into.
        bind_to_core (bool): Bind each MPI process to a core.
    """
    if n<=1: 
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += ['python3', '-m', _abs_path_2_module_name(sys.argv[0])] + sys.argv[1:]
        subprocess.check_call(args, env=env)
        sys.exit()

def msg(m, string='') -> None:
    print(('Message from %d: %s \t '%(MPI.COMM_WORLD.Get_rank(), string))+str(m))

def proc_id() -> int:
    """Get rank of calling process."""
    return MPI.COMM_WORLD.Get_rank()

def num_procs() -> int:
    """Count active MPI processes."""
    return MPI.COMM_WORLD.Get_size()

def broadcast(x: Union[torch.Tensor, np.ndarray, Any], root: int=0) -> None:
    """Broadcast the variable across the processes

    The varaible will be shared FROM the process identified
    by `root` to others. 
    
    :param x: the variable to be broadcasted
    :param root: the process ID of root
    """
    MPI.COMM_WORLD.Bcast(x, root=root)

def gather(x: Union[torch.Tensor, np.ndarray, Any], root: int=0) -> None:
    data = MPI.COMM_WORLD.gather(x,root=root)
    if proc_id() == root:
        return data
    else:
        return None

def mpi_avg(x: Union[torch.Tensor, np.ndarray, List, Any], base=1):
    """Average a scalar or vector over MPI processes."""
    dividor = max(_sum(base), 1)
    return _sum(x) / dividor
    
def mpi_statistics_scalar(x: List, with_min_and_max=False):
    """Get mean/std and optional min/max of scalar x across MPI processes.
    
    :param x: An array containing samples of the scalar to produce
        statistics for.
    :param with_min_and_max: If true, return min and max of x in 
        addition to mean and std.
    
    :return: mean, and the standard deviation
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = _sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = _sum(np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = _op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = _op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std
