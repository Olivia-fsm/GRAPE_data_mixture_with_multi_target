import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset
import os
import torch
os.environ['TRANSFORMERS_CACHE'] = '/scratch/sfan/huggingface_cache'

tknzr = tiktoken.get_encoding("gpt2")

import argparse
args_parser = argparse.ArgumentParser()
# DomainConfigArguments
args_parser.add_argument('--subset', default='cluster_1', type=str)


def get_climblab(subset='cluster_1', num_proc=40,
                       return_torch=False,):
    # {
    #     "tokens": ...,
    #     "token_count": ...,
    # }
    CLIMB_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/climblab/")
    SUBSET_PATH = os.path.join(CLIMB_DATA_PATH, subset)
    if not os.path.exists(os.path.join(SUBSET_PATH, 'val.bin')):
        # os.makedirs(SUBSET_PATH, exist_ok=True)
        # dataset = load_dataset("nvidia/ClimbLab", split=['train'])
        parquet_paths = [os.path.join(SUBSET_PATH,p) for p in os.listdir(SUBSET_PATH) if p.endswith("parquet")][:10]
        dataset = Dataset.from_parquet(parquet_paths)
        
        split_dataset = dataset.train_test_split(test_size=0.05, seed=2357, shuffle=True)
        split_dataset['val'] = split_dataset.pop('test')
        data_dict = {
            'train': split_dataset['train'],
            'val': split_dataset['val'],
        }

        
        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in data_dict.items():
            arr_len = np.sum(dset['token_count'])
            filename = os.path.join(SUBSET_PATH, f'{split}.bin')
            dtype = np.uint16 # (Unify data type to avoid errors)
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            total_batches = 100

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                # Batch together samples for faster write
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['tokens'])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    train_data = np.memmap(os.path.join(SUBSET_PATH, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(SUBSET_PATH, 'val.bin'), dtype=np.uint16, mode='r')
    print(f'Subset {subset}: train[{len(train_data)}] | val[{len(val_data)}]')
    if return_torch:
        train_data = torch.tensor(np.array(train_data, dtype=np.uint16))
        val_data = torch.tensor(np.array(val_data, dtype=np.uint16))
    return {'train': train_data, 'val': val_data}

# if __name__ == "__main__":
#     args = args_parser.parse_args()
#     get_climblab(subset=args.subset)