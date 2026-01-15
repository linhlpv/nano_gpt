import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset 


num_process = 8

num_process_load_dataset = num_process

# get the encoding of gpt2 from the tiktoken
encoding = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    dataset = load_dataset("openwebtext", num_proc=num_process_load_dataset)
    print(dataset)
    print(dataset["train"]['text'][0])

    split_dataset = dataset["train"].train_test_split(test_size=5e-4, seed=2112, shuffle=True)
    split_dataset["val"] = split_dataset.pop("test") # rename the test to val

    print(split_dataset)