import os
import torch
from torch.utils.data import DataLoader
from functools import partial

from tiny_vlm.model import TinyVLM
from tiny_vlm.data import VQADataset, collate_fn

DATA_PATH = "examples/toy_data.json"
CHECKPOINT_PATH = "checkpoints/tiny-vlm.pt"
EPOCHS = 3
BATCH_SIZE = 2
LR = 1e-3


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = TinyVLM().to(device)

    # only projector is trainable
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    num_image_tokens = model.vision_encoder.config.image_size // model.vision_encoder.config.patch_size
    num_image_tokens = num_image_tokens ** 2  # (224/32)^2 = 49

    dataset = VQADataset(
        DATA_PATH,
        model.image_processor,
        model.tokenizer,
        num_image_tokens,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_token_id=model.tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(trainable, lr=LR)

    model.train()
    model.vision_encoder.eval()
    model.lm.eval()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(pixel_values, input_ids, attention_mask, labels)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"  epoch {epoch} step {step+1} loss {loss.item():.4f}")

        print(f"Epoch {epoch} avg loss: {total_loss / len(loader):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"projector": model.projector.state_dict()}, CHECKPOINT_PATH)
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
