from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    """Builds model inputs from MathVQASample.

    The processor owns all text/image preprocessing that must be deterministic
    across train and inference.
    """

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    
    def _split_into_tiles(self, image: Image.Image) -> list[Image.Image]:
        num_tiles = max(1, self.config.num_tiles)
        width, height = image.size
        grid_size = 1
        while grid_size * grid_size < num_tiles:
            grid_size += 1
        tile_width = max(1, width // grid_size)
        tile_height = max(1, height // grid_size)
        tiles: list[Image.Image] = []

        for row in range(grid_size):
            for col in range(grid_size):
                if len(tiles) >= num_tiles:
                    break
                left = col * tile_width
                upper = row * tile_height
                right = width if col == grid_size - 1 else min(width, left + tile_width)
                lower = height if row == grid_size - 1 else min(height, upper + tile_height)
                tiles.append(image.crop((left, upper, right, lower)))
            if len(tiles) >= num_tiles:
                break

        return tiles
    

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert image to tensor with shape [num_tiles, 3, image_size, image_size].

        TODO:
            - convert to RGB;
            - resize/crop/pad;
            - split into tiles if num_tiles > 1;
            - normalize to float tensor.
        """
        image = image.convert("RGB")
        if self.config.num_tiles <= 1:
            tiles = [image]
        else:
            tiles = self._split_into_tiles(image)
        
        processed_tiles: list[torch.Tensor] = []
        for tile in tiles:
            tile = tile.resize(
                (self.config.image_size, self.config.image_size),
                Image.Resampling.BICUBIC,
            )
            pixel_values = torch.tensor(
                list(tile.getdata()),
                dtype=torch.float32,
            ).view(self.config.image_size, self.config.image_size, 3)
            pixel_values = pixel_values / 255.0
            pixel_values = pixel_values.permute(2, 0, 1).contiguous()
            processed_tiles.append(pixel_values)

        return torch.stack(processed_tiles, dim=0)
    

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        """Build a text prompt with visual special tokens and options.

        For training, include_answer=True should append the assistant answer.
        For inference, include_answer=False should stop before the answer.
        """
        image_tokens = " ".join(
            [IMAGE_TOKEN for _ in range(self.config.num_image_tokens)]
        )

        options_text = "\n".join(sample.options)

        prompt = (
            f"{IMAGE_START_TOKEN} {image_tokens} {IMAGE_END_TOKEN}\n"
            f"Question: {sample.question}\n"
            f"Options:\n"
            f"{options_text}\n"
            f"Answer:"
        )

        if include_answer:
            prompt = f"{prompt} {sample.answer}"

        return prompt
    

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask and labels for one sample.

        labels must be IGNORE_INDEX for prompt tokens and real token ids only
        for the assistant answer.
        """
        prompt = self.build_prompt(sample, include_answer=False)
        answer = f" {sample.answer}"
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            answer_ids = answer_ids + [eos_token_id]

        max_length = self.config.max_length
        if len(prompt_ids) + len(answer_ids) > max_length:
            max_prompt_length = max_length - len(answer_ids)

            if max_prompt_length > 0:
                prompt_ids = prompt_ids[:max_prompt_length]
            else:
                prompt_ids = []
                answer_ids = answer_ids[:max_length]

        input_ids = prompt_ids + answer_ids
        attention_mask = [1] * len(input_ids)
        labels = [self.config.ignore_index] * len(prompt_ids) + answer_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad text fields and stack pixel_values.

        TODO:
            - pad input_ids with tokenizer.pad_token_id;
            - pad attention_mask with 0;
            - pad labels with ignore_index;
            - stack pixel_values into [B, T, 3, H, W].
        """
        max_length = max(item["input_ids"].shape[0] for item in batch)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
        if pad_token_id is None:
            pad_token_id = 0
        input_ids_batch: list[torch.Tensor] = []
        attention_mask_batch: list[torch.Tensor] = []
        labels_batch: list[torch.Tensor] = []

        for item in batch:
            current_length = item["input_ids"].shape[0]
            pad_length = max_length - current_length
            input_ids_batch.append(
                F.pad(
                    item["input_ids"],
                    pad=(0, pad_length),
                    value=pad_token_id,
                )
            )
            attention_mask_batch.append(
                F.pad(
                    item["attention_mask"],
                    pad=(0, pad_length),
                    value=0,
                )
            )
            labels_batch.append(
                F.pad(
                    item["labels"],
                    pad=(0, pad_length),
                    value=self.config.ignore_index,
                )
            )

        return {
            "input_ids": torch.stack(input_ids_batch, dim=0),
            "attention_mask": torch.stack(attention_mask_batch, dim=0),
            "labels": torch.stack(labels_batch, dim=0),
            "pixel_values": torch.stack(
                [item["pixel_values"] for item in batch],
                dim=0,
            ),
        }

