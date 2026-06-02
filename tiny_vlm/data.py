import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class VQADataset(Dataset):
    def __init__(self, json_path, image_processor, tokenizer, num_image_tokens):
        with open(json_path) as f:
            self.samples = json.load(f)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.num_image_tokens = num_image_tokens
        self.data_dir = Path(json_path).parent

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self.data_dir / sample["image"]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        prompt = f"Question: {sample['question']}\nAnswer:"
        answer = f" {sample['answer']}"

        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + answer_ids
        # labels: -100 for prompt, actual ids for answer
        labels = [-100] * len(prompt_ids) + answer_ids

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch, pad_token_id):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    max_len = max(x["input_ids"].shape[0] for x in batch)

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, x in enumerate(batch):
        L = x["input_ids"].shape[0]
        input_ids[i, :L] = x["input_ids"]
        attention_mask[i, :L] = x["attention_mask"]
        labels[i, :L] = x["labels"]

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
