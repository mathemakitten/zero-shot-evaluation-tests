# zero-shot-evaluation-tests

### Notes
* Setup deep learning VM setup with Pytorch
* `conda install mpi4py` but not `pip install mpi4py`
* NCCL backend, not MPI.
* `transformers` uses 2x the model size to load into memory
  * CPU memory for 6.7B peaks around 50 GB
  * CPU memory for 13B peaks around 92 GB of memory 