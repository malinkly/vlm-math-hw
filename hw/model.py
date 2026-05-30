from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vision_hidden_size: int
    text_hidden_size: int
    num_image_tokens: int
    image_token_id: int


class VisionToTextAdapter(nn.Module):
    """Maps vision encoder hidden states to LLM embedding space."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        num_image_tokens: int,
    ) -> None:
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.num_image_tokens = num_image_tokens

        # TODO: replace with a small projection network.
        # Recommended: LayerNorm -> Linear -> GELU -> Linear.
        self.proj = nn.Sequential(
            nn.LayerNorm(vision_hidden_size),
            nn.Linear(vision_hidden_size, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, text_hidden_size),
        )


    def forward(self, vision_hidden_states: torch.Tensor) -> torch.Tensor:
        """Return visual embeddings [B, num_image_tokens, text_hidden_size]."""

        if vision_hidden_states.shape[1] != self.num_image_tokens:
            vision_hidden_states = F.adaptive_avg_pool1d(
                vision_hidden_states.transpose(1, 2),
                self.num_image_tokens,
            ).transpose(1, 2)

        return self.proj(vision_hidden_states)


def merge_visual_embeddings(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    visual_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Replace embeddings at <image> token positions with visual embeddings.

    Args:
        input_embeds: [B, L, D] text embeddings.
        input_ids: [B, L] token ids.
        visual_embeds: [B, K, D] visual embeddings.
        image_token_id: token id used as visual placeholder.

    Returns:
        Tensor [B, L, D] with visual embeddings inserted.

    Assumption for public tests:
        each row has exactly K positions where input_ids == image_token_id.
    """
    res = input_embeds.clone()
    for batch_idx in range(input_ids.shape[0]):
        image_positions = torch.nonzero(
            input_ids[batch_idx] == image_token_id,
            as_tuple=False,
        ).squeeze(-1)
        res[batch_idx, image_positions] = visual_embeds[batch_idx]

    return res


class MathVLM(nn.Module):
    """Thin wrapper around vision encoder, adapter and language model.

    In Track A/B, vision encoder and LLM should be frozen; adapter trainable.
    """

    def __init__(self, vision_encoder: nn.Module, language_model: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.config = config
        self.adapter = VisionToTextAdapter(
            vision_hidden_size=config.vision_hidden_size,
            text_hidden_size=config.text_hidden_size,
            num_image_tokens=config.num_image_tokens,
        )

    
    def freeze_backbones(self) -> None:
        """Freeze vision encoder and language model parameters."""
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.language_model.parameters():
            p.requires_grad = False
    

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run vision encoder and return hidden states [B, S, vision_hidden_size]."""
        if pixel_values.ndim == 5:
            batch_size, num_tiles, channels, height, width = pixel_values.shape
            flat_pixel_values = pixel_values.view(
                batch_size * num_tiles,
                channels,
                height,
                width,
            )
        else:
            batch_size = pixel_values.shape[0]
            num_tiles = 1
            flat_pixel_values = pixel_values

        vision_outputs = self.vision_encoder(flat_pixel_values)
        if hasattr(vision_outputs, "last_hidden_state"):
            hidden_states = vision_outputs.last_hidden_state
        elif isinstance(vision_outputs, dict) and "last_hidden_state" in vision_outputs:
            hidden_states = vision_outputs["last_hidden_state"]
        elif isinstance(vision_outputs, (tuple, list)):
            hidden_states = vision_outputs[0]
        else:
            hidden_states = vision_outputs

        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(1)
        if num_tiles > 1:
            hidden_states = hidden_states.view(
                batch_size,
                num_tiles,
                hidden_states.shape[1],
                hidden_states.shape[2],
            )
            hidden_states = hidden_states.reshape(
                batch_size,
                num_tiles * hidden_states.shape[2],
                hidden_states.shape[3],
            )

        return hidden_states


    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        """Forward pass with loss.

        TODO:
            - encode images;
            - map to visual embeddings;
            - get text input embeddings;
            - merge visual/text embeddings;
            - call language_model with inputs_embeds, attention_mask, labels.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")
        pixel_values = batch["pixel_values"]
        vision_hidden_states = self._encode_images(pixel_values)
        visual_embeds = self.adapter(vision_hidden_states)
        input_embedding_layer = self.language_model.get_input_embeddings()
        input_embeds = input_embedding_layer(input_ids)
        merged_embeds = merge_visual_embeddings(
            input_embeds=input_embeds,
            input_ids=input_ids,
            visual_embeds=visual_embeds,
            image_token_id=self.config.image_token_id,
        )

        return self.language_model(
            inputs_embeds=merged_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, batch: dict[str, torch.Tensor], **generation_kwargs: Any) -> torch.Tensor:
        """Generate answer token ids."""
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        pixel_values = batch["pixel_values"]

        vision_hidden_states = self._encode_images(pixel_values)
        visual_embeds = self.adapter(vision_hidden_states)

        input_embedding_layer = self.language_model.get_input_embeddings()
        input_embeds = input_embedding_layer(input_ids)

        merged_embeds = merge_visual_embeddings(
            input_embeds=input_embeds,
            input_ids=input_ids,
            visual_embeds=visual_embeds,
            image_token_id=self.config.image_token_id,
        )

        return self.language_model.generate(
            inputs_embeds=merged_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
