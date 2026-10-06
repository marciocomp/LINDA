# alpaca.py
# @author: Marcio Lopes
import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from utils.huggingface_token import HF_TOKEN
from utils.linda_logger import logger


class AlpacaDataset(Dataset):
    """
    Standard Alpaca Dataset for Instruction Tuning.
    Supports partitioning via 'indices' argument for Distributed Training.
    """

    def __init__(self,
                 tokenizer_name="google/gemma-2b",
                 max_length=512,
                 split="train",
                 indices=None,
                 cache_dir=None,
                 return_text_fields=False):
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.cache_dir = os.path.expanduser(cache_dir)
        self.return_text_fields = return_text_fields

        logger.info(f"Loading Tokenizer: {tokenizer_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=HF_TOKEN)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading Alpaca dataset from {self.cache_dir} (Split: {split})...")
        full_dataset = load_dataset("tatsu-lab/alpaca", split=split, cache_dir=self.cache_dir)

        if indices is not None:
            self.dataset = full_dataset.select(indices)
            logger.info(f"Dataset partitioned. Using {len(self.dataset)} samples based on provided indices.")
        else:
            self.dataset = full_dataset
            logger.info(f"Using full dataset ({len(self.dataset)} samples).")

    def __len__(self):
        return len(self.dataset)

    # ---------- Useful for generation evaluation ----------
    def get_prompt_and_reference(self, idx):
        item = self.dataset[idx]
        prompt = self._generate_prompt(item, include_output=False)
        reference = item["output"]
        return prompt, reference

    def __getitem__(self, idx):
        item = self.dataset[idx]

        prompt = self._generate_prompt(item, include_output=False)
        full_text = prompt + item["output"]

        # Tokenize full text (prompt + response)
        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Standard labels (Causal LM)
        labels = input_ids.clone()

        # 1) Mask padding
        labels[attention_mask == 0] = -100

        # 2) Mask the entire prompt (train/evaluate only on the response)
        prompt_ids = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )["input_ids"].squeeze(0)

        prompt_len = min(prompt_ids.numel(), self.max_length)
        labels[:prompt_len] = -100

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

        if self.return_text_fields:
            out["prompt_text"] = prompt
            out["reference_text"] = item["output"]

        return out

    # ---------- Prompt generation (with or without output) ----------
    def _generate_prompt(self, data_point, include_output=True):
        if data_point["input"]:
            base = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n"
                f"{data_point['instruction']}\n\n"
                "### Input:\n"
                f"{data_point['input']}\n\n"
                "### Response:\n"
            )
        else:
            base = (
                "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n"
                f"{data_point['instruction']}\n\n"
                "### Response:\n"
            )

        if include_output:
            return base + f"{data_point['output']}"
        return base


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    labels[attention_mask == 0] = -100
    return input_ids, attention_mask, labels
