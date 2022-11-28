
import numpy as np
from typing import Dict, Optional, Tuple
import os
import json
from random import shuffle
import time
import sys
import logging
import transformers
from transformers.file_utils import ModelOutput
import torch.nn as nn
import torch
from transformers.activations import gelu
from transformers import LxmertConfig

from .config import VisionTapasConfig
from .vision_tapas import VisionTapasPreTrainedModel, VisionTapasModel

class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

class VisionTapasForClassificationOutput(ModelOutput):
    """
    VisionTapasModelOutput's outputs that contain the last hidden states, pooled outputs, and attention probabilities for the language,
    visual, and, cross-modality encoders. (note: the visual encoder in Lxmert is referred to as the "relation-ship"
    encoder")
    """

    loss: Optional[torch.FloatTensor] = None
    tapas_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vit_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    tapas_attentions: Optional[Tuple[torch.FloatTensor]] = None
    vit_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None



class VisionTapasClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        hid_dim = hidden_size
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            nn.LayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_labels),
        )

    def forward(self, pooled_output):
        return self.logit_fc(pooled_output)


class VisionTapasForClassification(VisionTapasPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Configuration
        self.config = config
        self.num_labels = config.num_labels

        # VisionTaPas Model
        self.visiontapas = VisionTapasModel(config)

        # Classification head
        self.classifier = VisionTapasClassificationHead(self.visiontapas.lxmert_config.hidden_size, self.num_labels)

        # Loss function
        self.loss = nn.CrossEntropyLoss()

        # Initialize weights
        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask, pixel_values=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        r"""
        labels: (``Torch.Tensor`` of shape ``(batch_size)``, `optional`):
            A one-hot representation of the correct answer

        Returns:
        """


        visiontapas_output = self.visiontapas(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        pooled_output = visiontapas_output.pooled_output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))

        all_hidden_states = ()
        if output_hidden_states:
            all_hidden_states = (visiontapas_output.tapas_hidden_states, visiontapas_output.vit_hidden_states)

        all_attentions = ()
        if output_attentions:
            all_attentions = (visiontapas_output.tapas_attentions, visiontapas_output.vit_attentions, visiontapas_output.cross_encoder_attentions)

        if not return_dict:
            if labels is not None:
                return (loss, logits)
            else:
                return (logits)

        return VisionTapasForClassificationOutput(
            loss=loss,
            logits=logits,
            tapas_hidden_states=visiontapas_output.tapas_hidden_states,
            vit_hidden_states=visiontapas_output.vit_hidden_states,
            tapas_attentions=visiontapas_output.tapas_attentions,
            vit_attentions=visiontapas_output.vit_attentions,
            cross_encoder_attentions=visiontapas_output.cross_encoder_attentions,
        )


