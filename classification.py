# deepspeed --num_gpus 8 classification.py

import os

import ast
import time
import deepspeed
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.deepspeed import HfDeepSpeedConfig

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

# distributed setup
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()


def load_dataset(num_truncate=None):
    df = pd.read_csv('conjunction_fallacy-0shot.csv').iloc[:, 1:]  # drop nonsensical 1st column
    if num_truncate:
        df = df[:num_truncate]
        print(f"Running with dataset truncated to {num_truncate} examples")
    examples = df.to_dict('records')

    prompts = [
        example["prompt"] + class_seq
        for example in examples
        for class_seq in ast.literal_eval(example["classes"])
    ]
    return prompts


def prepare(dataset, batch_size=8, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size,
                                 rank=torch.distributed.get_rank(),
                                 shuffle=False,
                                 drop_last=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                            drop_last=False, shuffle=False, sampler=sampler)

    return dataloader


class Inference:

    def __init__(self):

        self.model_name = 'facebook/opt-6.7b'
        self.model_name = "facebook/opt-13b"
        self.model_name = "facebook/opt-30b"
        self.model_name = "facebook/opt-66b"

        # FYI: from_pretrained(..., low_cpu_mem_usage=True) is incompatible with zero.init stage 3
        #  normally Transformers uses 2x of model size in CPU memory while loading the model
        self.config = AutoConfig.from_pretrained(self.model_name)
        # print(config)
        self.model_hidden_size = self.config.hidden_size

        self.train_batch_size = 1 * world_size

        ds_config = {
            "fp16": {
                "enabled": False,
            },
            "bf16": {
                "enabled": False,
            },
            "zero_optimization": {
                "stage": 3,
                "offload_param": {
                    "device": "cpu",  # change this from "none" to offload to CPU: {"device": "cpu"}
                    # "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": self.model_hidden_size * self.model_hidden_size,
                "stage3_prefetch_bucket_size": 0.9 * self.model_hidden_size * self.model_hidden_size,
                "stage3_param_persistence_threshold": 10 * self.model_hidden_size
            },
            "steps_per_print": 2000,
            "train_batch_size": self.train_batch_size,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": True
        }

        # keep this object alive since we're not using transformers Trainer
        #  see: https://huggingface.co/docs/transformers/main/main_classes/deepspeed#deepspeed-non-trainer-integration
        self.dschf = HfDeepSpeedConfig(ds_config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # we are ready to initialise deepspeed ZeRO now
        self.ds_engine = deepspeed.initialize(model=self.model,  # this should really be init_inference()
                                              config_params=ds_config,
                                              model_parameters=None,
                                              optimizer=None,
                                              lr_scheduler=None)[0]
        self.ds_engine.module.eval()  # inference

    def evaluate_classification(self, distributed_dataset):

        batch_id = 0

        while True:  # TODO: change this to be a for loop
            try:
                batch = next(distributed_dataset)
                option0, option1 = batch[::2], batch[1::2]  # TODO: change dist dataset to zip as tuple or dict

                batch_id += 1
                print(f"batch id: {batch_id}")

                # TODO: document why this; check if distributed data loader can be constructed w tuples/dicts
                #  also figure out if local_rank here should be torch.distributed.get_rank()
                tokenized_inputs0 = self.tokenizer(option0, return_tensors="pt", padding=True).to(device=local_rank)
                tokenized_inputs1 = self.tokenizer(option1, return_tensors="pt", padding=True).to(device=local_rank)
                # tokenized_inputs0 = self.tokenizer(option0, return_tensors="pt", padding=True).to(device=torch.distributed.get_rank())
                # tokenized_inputs1 = self.tokenizer(option1, return_tensors="pt", padding=True).to(device=torch.distributed.get_rank())
                outputs0 = self.ds_engine.module(**tokenized_inputs0)["logits"].detach().to(device="cpu", dtype=torch.float32)
                outputs1 = self.ds_engine.module(**tokenized_inputs1)["logits"].detach().to(device="cpu", dtype=torch.float32)

                # TODO: actual evaluation logic with logprobs; needs to handle padding in the batch bc elements
                #  in the batch are of different lengths
                #  but we're mostly interested in how fast we can pump data through for inference
                #  evaluation can happen later by taking logits off of the GPU

            except StopIteration:
                # print("exhausted iterator")
                return


if __name__ == '__main__':

    NUM_TRUNCATE = None  # 16

    # FYI: 1007 examples in the dataset, batch size of 16
    ds = load_dataset(num_truncate=NUM_TRUNCATE)
    ds_distributed = prepare(ds, batch_size=16)  # this is per-accelerator batch size
    ds_distributed_iter = iter(ds_distributed)
    #  note: this means len(next(ds_distributed_iter)) = 16, which is actually a batch size of 8 * 2 options
    #   distributed dataset turns this into 8 examples * 8 GPUs = 64 actual batch size
    #  1107 // 64 = 17 batches to run through the model

    st = time.time()
    infer = Inference()

    print(f"Compile time: {time.time() - st}")
    result = infer.evaluate_classification(ds_distributed_iter)
    print(f"Total time taken: {time.time() - st}")
