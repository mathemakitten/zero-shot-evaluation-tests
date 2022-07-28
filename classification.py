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


# def get_logits_and_tokens(self, prompts):
#     all_logits = []
#     all_tokens = []
#
#     for prompt in prompts:
#         tokenized_inputs = self.tokenizer(
#             prompt, return_tensors="pt", truncation=True
#         ).to(device=local_rank)
#         outputs = self.ds_engine.module(**tokenized_inputs)
#         logits = outputs["logits"].detach().to(device="cpu", dtype=torch.float32)
#
#         # need to remove batch dimension
#         all_logits.append(torch.squeeze(logits))
#         all_tokens.append(torch.squeeze(tokenized_inputs["input_ids"]))
#     return all_logits, all_tokens


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
        # self.model_name = "facebook/opt-13b"

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
                    "device": "none",
                    "pin_memory": True
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
        self.ds_engine = deepspeed.initialize(model=self.model,
                                              config_params=ds_config,
                                              model_parameters=None,
                                              optimizer=None,
                                              lr_scheduler=None)[0]
        self.ds_engine.module.eval()  # inference

    def evaluate_classification(self, distributed_dataset):

        all_logits = []
        all_tokens = []

        batch_id = 0

        while True:
            try:
                batch = next(distributed_dataset)
                option0, option1 = batch[::2], batch[1::2]

                batch_id += 1
                print(f"batch id: {batch_id}")

                tokenized_inputs0 = self.tokenizer(option0, return_tensors="pt", padding=True).to(device=local_rank)
                tokenized_inputs1 = self.tokenizer(option1, return_tensors="pt", padding=True).to(device=local_rank)
                outputs0 = self.ds_engine.module(**tokenized_inputs0)["logits"].detach().to(device="cpu", dtype=torch.float32)
                outputs1 = self.ds_engine.module(**tokenized_inputs1)["logits"].detach().to(device="cpu", dtype=torch.float32)

                # TODO: actual evaluation logic with logprobs; needs to handle padding in the batch bc elements
                #  in the batch are of different lengths
                #  but we're mostly interested in how fast we can pump data through for inference
                #  evaluation can happen later by taking logits off of the GPU

                # logits = outputs["logits"].detach().to(device="cpu", dtype=torch.float32)

                # for x in batch:
                #     tokenized_inputs = self.tokenizer(x, return_tensors="pt", truncation=True).to(device=local_rank)
                #     outputs = self.ds_engine.module(**tokenized_inputs)
                #     logits = outputs["logits"].detach().to(device="cpu", dtype=torch.float32)
                #
                #     # need to remove batch dimension
                #     all_logits.append(torch.squeeze(logits))
                #     all_tokens.append(torch.squeeze(tokenized_inputs["input_ids"]))
            except StopIteration:
                # print("exhausted iterator")
                return #{"all_logits": all_logits, "all_tokens": all_tokens}

         # TODO: run evaluation here after inference


if __name__ == '__main__':

    NUM_TRUNCATE = None  # 16

    # FYI: 1007 examples, batch size of 16
    ds = load_dataset(num_truncate=NUM_TRUNCATE)
    ds_distributed = prepare(ds, batch_size=16)  # this is per-accelerator batch size
    ds_distributed_iter = iter(ds_distributed)
    #  note: this means len(next(ds_distributed_iter)) = 16, which is actually a batch size of 8 * 2 options
    #   distributed dataset turns this into 8 * 8 = 64 actual batch size
    #  1107 // 64 = 17 batches to run through the model

    # exit()

    st = time.time()
    infer = Inference()
    result = infer.evaluate_classification(ds_distributed_iter)

    print(f"Time taken: {time.time() - st}")
