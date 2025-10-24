# ipy-slurm-exec

Jupyter Notebook + Slurm integration, not just an interface.

## Explain by example

### Import
```
import ipy_slurm_exec
%load_ext ipy_slurm_exec
```

### Create data in Notebook

```
import numpy as np
vector = np.linspace(0, 9, 10)
```

### Submit a calculation to Slurm cluster
```
%%slurm_exec gpu_result vector --partition=gpu-turing --gpus=1 --modules=python-cbrg,cuda
import cupy as cp
device_vec = cp.asarray(vector)
gpu_result = cp.asnumpy(device_vec ** 2)
```

### Wait for job to complete
```
Submitted Slurm job 4382780 (folder: /ceph/project/sysadmin/aowenson/Jupyter/NetworkMap/slurm_jobs/slurm-exec-20251024-135830-89596489)
Job 4382780 completed. Result assigned to 'gpu_result'
```

### Result in Notebook
```
print(gpu_result)
[ 0.  1.  4.  9. 16. 25. 36. 49. 64. 81.]
```
