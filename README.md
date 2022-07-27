# zero-shot-evaluation-tests

### Notes
* Setup deep learning VM setup with Pytorch
* `conda install mpi4py` but not `pip install mpi4py`
* NCCL backend, not MPI.
* CPU memory for 6.7B peaking around ~50GB (makes sense as `transformers` uses 2x the model size to load into memory)