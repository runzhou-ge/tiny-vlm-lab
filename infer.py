import argparse
import torch
from PIL import Image

from tiny_vlm.model import TinyVLM

CHECKPOINT_PATH = "checkpoints/tiny-vlm.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--question", required=True, help="Question about the image")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyVLM().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.projector.load_state_dict(ckpt["projector"])
    model.eval()

    image = Image.open(args.image).convert("RGB")
    answer = model.generate(image, args.question, device=device)
    print(answer)


if __name__ == "__main__":
    main()
