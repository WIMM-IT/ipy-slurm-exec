# ipy-slurm-exec

Jupyter Notebook + Slurm integration.

```
pip install ipy-slurm-exec
```

![Simple example](https://github.com/WIMM-IT/ipy-slurm-exec/raw/main/docs/demo.gif)

## `slurm_exec` arguments:

Specify arguments with `%%slurm_exec` to manage variables and the Slurm job.

### Managing variables

If you do not specify a list of variables to export, then all are exported to the Slurm job.
Similarly for importing after the job finishes.
For large notebooks this may cause problems - 
overwriting a variable in another part of your notebook, or exporting many big variables that are never used (wasting memory).
Use these arguments to manage.

```
-i, --inputs: list of variables to input into the Slurm job

e.g. %%slurm_exec -i data,scale ...
```

```
-o, --outputs: list of output variables to import into Notebook from Slurm job

e.g. %%slurm_exec -o reduced ...
```

#### Import fail

Python variables created in the Slurm job may be dependent on additional modules it loaded that were not loaded by your Notebook, for example CUDA.
These cannot be imported into Notebook so will be skipped - the execution report will list when this happens e.g.:

```
Skipped variables in Notebook:
  device_vec: 'cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version'
```


### Slurm job parameters

Refer to any Slurm cluster documentation for setting these parameters, as the values are simply passed to Slurm.
When not set then Slurm defaults are used, same as when directly submitting a job to Slurm.

```
--partition
--time
--ntasks
--cpus-per-task
--mem
--gpus

e.g. %%slurm_exec --partition=gpu --gpus=1 --time=00:10:00
```

### GNU environment modules

```
--modules: list of modules to load in Slurm job. Add prefix '+' to have job inherit modules loaded in Notebook.

e.g. %%slurm_exec ... --modules=+cuda
```

## GPU example

Bringing everything together, here is a more realistic example - running a Notebook cell on a GPU.

Import `ipy_slurm_exec`
```
import ipy_slurm_exec
%load_ext ipy_slurm_exec
```

Setup data in Notebook:
```
import torch
import numpy as np
seed = 123
vector = np.linspace(-2, 2, 256, dtype=np.float32)
torch.manual_seed(seed)
```

Code to run on a GPU:
```
%%slurm_exec -i seed,vector -o torch_result --partition=gpu --gpus=1 --mem=1G
torch.manual_seed(seed)
device = torch.device("cuda")
x = torch.from_numpy(vector).to(device)
y = torch.tanh(x @ x.T)
torch_result = y.sum().item()
```

Execution report:
```
Submitted Slurm job 1326 (folder: slurm_exec/20251223T1152-dc0f96dc)
Job completed                                                                   
Imported: torch_result
```
