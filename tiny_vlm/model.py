import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()


VISION_MODEL_ID = "openai/clip-vit-base-patch32"
LANGUAGE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_processor = CLIPImageProcessor.from_pretrained(VISION_MODEL_ID)
        self.vision_encoder = CLIPVisionModel.from_pretrained(VISION_MODEL_ID)
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_ID)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(LANGUAGE_MODEL_ID)
        for p in self.lm.parameters():
            p.requires_grad = False

        vision_dim = self.vision_encoder.config.hidden_size
        lm_dim = self.lm.config.hidden_size
        self.projector = nn.Linear(vision_dim, lm_dim)

    def encode_image(self, pixel_values):
        with torch.no_grad():
            outputs = self.vision_encoder(pixel_values=pixel_values.to(self.vision_encoder.dtype))
        # patch tokens only (drop CLS), shape: (B, num_patches, vision_dim)
        image_features = outputs.last_hidden_state[:, 1:, :]
        lm_dtype = next(self.lm.parameters()).dtype
        return self.projector(image_features.to(self.projector.weight.dtype)).to(lm_dtype)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        image_embeds = self.encode_image(pixel_values)  # (B, N_img, lm_dim)
        num_image_tokens = image_embeds.shape[1]

        text_embeds = self.lm.get_input_embeddings()(input_ids)  # (B, T, lm_dim)

        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        # extend attention mask for image tokens
        B = pixel_values.shape[0]
        img_attn = torch.ones(B, num_image_tokens, device=attention_mask.device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([img_attn, attention_mask], dim=1)

        if labels is not None:
            img_labels = torch.full((B, num_image_tokens), -100, device=labels.device, dtype=labels.dtype)
            full_labels = torch.cat([img_labels, labels], dim=1)
        else:
            full_labels = None

        return self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=128, device="cpu"):
        prompt = f"Question: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        image_embeds = self.encode_image(pixel_values)  # (1, N_img, lm_dim)
        num_image_tokens = image_embeds.shape[1]

        text_embeds = self.lm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        img_attn = torch.ones(1, num_image_tokens, device=device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([img_attn, attention_mask], dim=1)

        out = self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
