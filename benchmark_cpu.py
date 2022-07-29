# How fast can you pipe this dataset through on CPU only? Computed on n1-standard-8, w 8x CPUs and 30 GB RAM
import pandas as pd
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time

# Just so we can have batched data for free
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


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


ds = load_dataset(num_truncate=None)  # 276 batches

model_name = "facebook/opt-6.7b"  # requires 35 GB of RAM to load
model_name = "facebook/opt-13b"

conf = AutoConfig.from_pretrained(model_name, use_auth_token=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name)

st = time.time()
for i, x in enumerate(list(chunks(ds, n=8))):
    print(f"Batch {i}")
    # Tokenize
    tokenized_inputs = tokenizer(x, return_tensors="pt", padding=True)
    outputs = model(**tokenized_inputs)

print(f"Time taken, without compilation: {time.time() - st}")
