# ipy-slurm-exec

Jupyter Notebook + Slurm integration.

## Quick start

Setup data in Notebook:
```
import numpy as np
data = np.arange(12).reshape(3, 4)
scale = 2.5
```

Use `%%slurm_exec` cell magic to setup code to run Slurm cluster (default queue and resources):
```
%%slurm_exec
import numpy as np
scaled = np.asarray(data) * scale
reduced = scaled.sum(axis=0)
```

Execution report:
```
Submitted Slurm job 4394414 (folder: slurm_exec/20251028T1620-4377f370)
Job completed                                                                   
Updated variables: data, params, reduced, scaled
```

Print result in Notebook:
```
print(reduced)
[30.  37.5 45.  52.5]
```

## `slurm_exec` arguments:

Specify arguments after `%%slurm_exec` to manage variables and the Slurm job.

### Managing variables

If you do not specify a list of variables to export, then all are exported to the Slurm job.
Similarly for importing after the job finishes.
For small notebooks this probably does not matter, 
but for large notebooks this may cause problems - 
overwriting a variable in another part of your notebook, or exporting many big variables that are never used (wasting memory).
So use these arguments to manage.

```
-i, --inputs: list of variables to input into the Slurm job

e.g. %%slurm_exec -i data,scale ...
```

```
-o, --outputs: list of output variables to import into Notebook from Slurm job

e.g. %%slurm_exec -o reduced ...
```

In a large or complex Notebook, you probably want to carefully manage which variables are exported and imported. 

#### Import fail

When you do not specify `-o` argument, then `slurm_exec` will attempt to import all variables from the Slurm job. 
However some may be dependent on additional modules loaded by job that were not loaded by your Notebook, for example CUDA, so cannot be imported.
The execution report will list variables that could not be imported with reason why, e.g.:

```
Skipped variables in Notebook:
  device_vec: 'cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version'
```


### Slurm job parameters

```
--partition
--time
--ntasks
--cpus-per-task
--mem
--gpus

e.g. %%slurm_exec --partition=gpu --gpus=1 --time=00:10:00
```

Refer to any Slurm cluster documentation for setting these parameters, as the values are simply passed to Slurm.

### GNU environment modules

```
--modules: list of modules to load in Slurm job. Add prefix '+' to have job inherit modules loaded in Notebook.

e.g. %%slurm_exec ... --modules=+cuda
```

## GPU example

Bringing everything together, here is a more realistic example - running a Notebook cell on a GPU.

Setup data in Notebook:
```
import numpy as np
vector = np.linspace(0, 9, 10)
```

Code to run on a GPU:
```
%%slurm_exec -i vector -o gpu_result --partition=gpu --gpus=1 --time=00:10:00 --modules=+cuda
import cupy as cp
device_vec = cp.asarray(vector)
gpu_result = cp.asnumpy(device_vec ** 2)
```

Execution report:
```
Submitted Slurm job 4394748 (folder: slurm_exec/20251028T1650-e3c49805)
Job completed                                                                   
Updated variables: gpu_result
{'gpu_result': array([ 0.,  1.,  4.,  9., 16., 25., 36., 49., 64., 81.])}
```
