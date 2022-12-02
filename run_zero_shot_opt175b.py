import ast
import deepspeed
import inspect
import json
import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.deepspeed import HfDeepSpeedConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

# distributed setup
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()

train_batch_size = 1 * world_size

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
            "device": "cpu",  # Set to None or "cpu"
            # "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
        "stage3_param_persistence_threshold": 10 * model_hidden_size
    },
    "steps_per_print": 2000,
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
}

dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive

model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
print(f"Model initialized: {model}")

# we are ready to initialise deepspeed ZeRO now
ds_engine = deepspeed.initialize(model=model,
                                 config_params=ds_config,
                                 model_parameters=None,
                                 optimizer=None,
                                 lr_scheduler=None)[0]
ds_engine.module.eval()  # inference


def load_dataset(num_truncate=None):

    prompts = []  # list of dicts

    with open('winobias_antistereotype_test.jsonl', 'r') as json_file:
        json_list = list(json_file)

    examples = []

    for json_str in json_list:
        result = json.loads(json_str)
        examples.append(result)

    for e in examples:
        prompts.append({'example0': e['text'] + e['classes'][0], 'example1': e['text'] + e['classes'][1], 'answer': e['target']})

    return prompts


def prepare(dataset, batch_size=8, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size,
                                 rank=torch.distributed.get_rank(),
                                 shuffle=False,
                                 drop_last=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                            drop_last=False, shuffle=False, sampler=sampler)

    return dataloader


def get_logits_and_tokens(ds_engine, tokenizer, prompts):
    all_logits = []
    all_tokens = []
    for prompt in prompts:
        print(f"prompt: {prompt}")
        tokenized_inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True
        ).to(device=local_rank)
        outputs = ds_engine.module(**tokenized_inputs)
        logits = outputs["logits"].detach().to(device="cpu", dtype=torch.float32)
        # need to remove batch dimension
        all_logits.append(torch.squeeze(logits))
        all_tokens.append(torch.squeeze(tokenized_inputs["input_ids"]))
    return all_logits, all_tokens


class Inference:

    def __init__(self):

        self.model_name = 'facebook/opt-6.7b'
        self.model_name = "facebook/opt-13b"
        self.model_name = "facebook/opt-30b"
        self.model_name = "facebook/opt-66b"
        self.model_name = 'facebook/opt-125m'
        self.model_name = "huggingface/opt-175b"

        # FYI: from_pretrained(..., low_cpu_mem_usage=True) is incompatible with zero.init stage 3
        #  normally Transformers uses 2x of model size in CPU memory while loading the model
        self.config = AutoConfig.from_pretrained(self.model_name, use_auth_token=True)
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

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False, use_auth_token=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, use_auth_token=True)

        # we are ready to initialise deepspeed ZeRO now
        self.ds_engine = deepspeed.initialize(model=self.model,  # this should really be init_inference()
                                              config_params=ds_config,
                                              model_parameters=None,
                                              optimizer=None,
                                              lr_scheduler=None)[0]
        self.ds_engine.module.eval()  # inference

    def evaluate_classification(self, distributed_dataset):

        batch_id = 0
        num_total, num_correct = 0, 0

        while True:  # TODO: change this to be a for loop
            st = time.time()
            try:
                batch = next(distributed_dataset)
                num_total += 1
                # option0, option1 = batch[::2], batch[1::2]  # TODO: change dist dataset to zip as tuple or dict
                option0, option1 = batch['example0'], batch['example1']

                # print("data in")
                # print(option0)
                # print(option1)


                batch_id += 1
                # print(f"batch id: {batch_id}")

                # TODO: document why this; check if distributed data loader can be constructed w tuples/dicts
                #  also figure out if local_rank here should be torch.distributed.get_rank()
                tokenized_inputs0 = self.tokenizer(option0, return_tensors="pt", padding=True).to(device=local_rank)
                tokenized_inputs1 = self.tokenizer(option1, return_tensors="pt", padding=True).to(device=local_rank)

                # print("tokenized")
                # print(tokenized_inputs0)
                # print(tokenized_inputs1)

                # tokenized_inputs0 = self.tokenizer(option0, return_tensors="pt", padding=True).to(device=torch.distributed.get_rank())
                # tokenized_inputs1 = self.tokenizer(option1, return_tensors="pt", padding=True).to(device=torch.distributed.get_rank())
                outputs0 = self.ds_engine.module(**tokenized_inputs0)["logits"]#.detach().to(device="cpu", dtype=torch.float32)
                outputs1 = self.ds_engine.module(**tokenized_inputs1)["logits"]#.detach().to(device="cpu", dtype=torch.float32)

                # print("WHYyyy")
                # print(outputs0)

                # turn logits into logprobs
                logits0 = F.log_softmax(outputs0, dim=-1)
                logits1 = F.log_softmax(outputs1, dim=-1)

                # print(f"WHY. shape of logits0: {logits0.shape} logits1: {logits1.shape}")
                # print(logits0[0])
                # print(logits1[0])

                # sum the logprobs
                logprobs0, logprobs1 = 0.0, 0.0
                for t in range(tokenized_inputs0['input_ids'].shape[1] - 1):
                    # print(f"t: {t}")
                    # print(f"tokenized input: {tokenized_inputs0}")
                    # print(logits0.shape)
                    # print('here')
                    # print(logits0[0][t+1][tokenized_inputs0['input_ids'][0][t]].detach().to(device='cpu'))
                    logprobs0 += logits0[0][t+1][tokenized_inputs0['input_ids'][0][t]].detach().to(device='cpu')
                for t in range(tokenized_inputs1['input_ids'].shape[1] - 1):
                    # print('here2')
                    # print(logits1[0][t+1][tokenized_inputs1['input_ids'][0][t]].detach().to(device='cpu'))
                    logprobs1 += logits1[0][t+1][tokenized_inputs1['input_ids'][0][t]].detach().to(device='cpu')
                logprobs = [logprobs0.detach().to(device='cpu'), logprobs1.detach().to(device='cpu')]
                # print(f"logprobs: {logprobs}")
                chosen_ending = torch.argmax(torch.Tensor(logprobs)).detach().to(device="cpu")

                if num_total % 100 == 0:
                    print(f"iteration {num_total}")

                # print(f"chosen ending: {chosen_ending} answer: {batch['answer']}")

                if chosen_ending == batch['answer']:
                    num_correct += 1

                # if num_total == 200:
                #     print(f"num correct: {num_correct} num total: {num_total}")
                    # exit()

            except StopIteration:
                print(f"Took {time.time() - st} s")
                print(f"num correct: {num_correct} num total: {num_total} % correct: {num_correct / num_total}")
                # print("exhausted iterator")
                return


if __name__ == '__main__':

    NUM_TRUNCATE = None

    ds = load_dataset(num_truncate=NUM_TRUNCATE)
    ds_distributed = prepare(ds, batch_size=1)  # this is per-accelerator batch size
    ds_distributed_iter = iter(ds_distributed)

    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    # print(evaluate_classification(ds_engine, tokenizer, test_examples))

    st = time.time()
    infer = Inference()

    print(f"Compile time: {time.time() - st}")
    result = infer.evaluate_classification(ds_distributed_iter)
    print(f"Total time taken: {time.time() - st}")