# deepspeed --num_gpus 8 classification.py

import os

import time
import deepspeed
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.deepspeed import HfDeepSpeedConfig


class Inference:

    def __init__(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

        # distributed setup
        self.local_rank = int(os.getenv('LOCAL_RANK', '0'))
        self.world_size = int(os.getenv('WORLD_SIZE', '1'))
        torch.cuda.set_device(self.local_rank)
        deepspeed.init_distributed()

        # model_name = 'facebook/opt-6.7b'
        self.model_name = "facebook/opt-13b"
        # FYI: from_pretrained(..., low_cpu_mem_usage=True) is incompatible with zero.init stage 3
        #  normally Transformers uses 2x of model size in CPU memory while loading the model
        self.config = AutoConfig.from_pretrained(self.model_name)
        # print(config)
        self.model_hidden_size = self.config.hidden_size

        self.train_batch_size = 1 * self.world_size

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
            "wall_clock_breakdown": False
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

    def load_dataset(self, num_truncate=None):
        df = pd.read_csv('conjunction_fallacy-0shot.csv').iloc[:, 1:]  # drop nonsensical 1st column
        if num_truncate:
            df = df[:num_truncate]
        examples = df.to_dict('records')
        return examples

    def get_logits_and_tokens(self, prompts):
        all_logits = []
        all_tokens = []

        for prompt in prompts:
            tokenized_inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True
            ).to(device=self.local_rank)
            outputs = self.ds_engine.module(**tokenized_inputs)
            logits = outputs["logits"].detach().to(device="cpu", dtype=torch.float32)

            # need to remove batch dimension
            all_logits.append(torch.squeeze(logits))
            all_tokens.append(torch.squeeze(tokenized_inputs["input_ids"]))
        return all_logits, all_tokens

    def evaluate_classification(self, examples):
        prompts = [
            example["prompt"] + class_seq
            for example in examples
            for class_seq in example["classes"]
        ]

        all_logits, all_tokens = self.get_logits_and_tokens(prompts)

        # for each possible class sequence, we need to get the logprob on the full class sequence
        n_classes = len(examples[0]["classes"])
        total_logprobs = []
        losses = []
        labels_correct = []

        for i, example in enumerate(examples):
            prompt_start = i * n_classes
            class_logprobs = []
            for j in range(n_classes):
                class_index = prompt_start + j
                class_logits = all_logits[class_index]
                # the lengths of each class sequence in tokens
                class_sequence = example["classes"][j]
                target_token_length = len(self.tokenizer(class_sequence)["input_ids"])
                # we only need the logits for the end sequence
                tokens = all_tokens[class_index]
                # we have to go back by one because we don't care about the logits for the predicted token
                sequence_logits = class_logits[-target_token_length - 1: -1]
                sequence_tokens = tokens[-target_token_length:]
                # we take a log_softmax over all token logits for each position in the class sequence to
                #  get log probabilities, and then sum the logprobs for the tokens actually chosen
                logprobs = F.log_softmax(sequence_logits, dim=-1)
                class_logprob = sum(
                    [logprobs[i, token] for i, token in enumerate(sequence_tokens)]
                )
                class_logprobs.append(class_logprob.item())  # the sum is never empty so never just 0, always a tensor

            total_logprob = sum(class_logprobs)
            normalised_logprobs = F.log_softmax(torch.tensor(class_logprobs), dim=-1)
            loss = -normalised_logprobs[example["answer_index"]].item()
            label_correct = int(np.argmax(normalised_logprobs) == example["answer_index"])
            total_logprobs.append(total_logprob)
            losses.append(loss)
            labels_correct.append(label_correct)
        return {
            "loss": losses,
            "correct": labels_correct,
            "total_logprob": total_logprobs,
            "accuracy": sum(labels_correct) / len(labels_correct),
            "mean_loss": sum(losses) / len(losses)
        }


if __name__ == '__main__':
    infer = Inference()
    dataset = infer.load_dataset(num_truncate=10)
    result = infer.evaluate_classification(dataset)
    print(result)


test_examples = [
    {
        "prompt": """Question: Which is more likely?
A. Andrew is a scientist and is smart.
B. Andrew is a scientist.
Answer:""",
        "classes": [" A", " B"],
        "answer_index": 1
    },
    {
        "prompt": """Q: Which is more likely?
1. Michael is an accountant.
2. Michael is an accountant and is careful.
A:""",
        "classes": [" 1", " 2"],
        "answer_index": 0
    },
    {
        "prompt": """Q: Which is more likely to be true?
1. Jessica is a teacher.
2. Jessica is a teacher and is patient.
A:""",
        "classes": [" 1", " 2"],
        "answer_index": 0
    }
]