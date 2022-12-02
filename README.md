# zero-shot-evaluation-tests

```
# new conda env 

nvcc -V 

pip3 install torch==1.10.1+cu113 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install deepspeed
```

### Notes
* Setup deep learning VM setup with Pytorch
* `conda install mpi4py` but not `pip install mpi4py`
* NCCL backend, not MPI.
* `transformers` uses 2x the model size to load into memory
  * CPU memory for 6.7B peaks around 50 GB
  * CPU memory for 13B peaks around 92 GB of memory 