import numpy as np
from typing import Dict, Optional, Tuple
import os
import json
from random import shuffle
import time
import sys
import logging
import transformers
from transformers import PreTrainedModel, ViTModel, ViTFeatureExtractor, LxmertXLayer, TapasModel, TapasForSequenceClassification
from transformers.file_utils import ModelOutput
from transformers import TapasConfig, LxmertConfig, ViTConfig, ViTPreTrainedModel, PreTrainedModel
from transformers.models.vit.modeling_vit import ViTEncoder
import torch.nn as nn
from transformers import TapasTokenizer
import torch
import pandas as pd
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
from transformers import AdamW
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import EvalPrediction
from transformers import Trainer, TrainingArguments
import collections
from transformers.activations import gelu

from .config import VisionTapasConfig

class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)



class VisionTapasModelOutput(ModelOutput):
    """
    VisionTapasModelOutput's outputs that contain the last hidden states, pooled outputs, and attention probabilities for the language,
    visual, and, cross-modality encoders. (note: the visual encoder in Lxmert is referred to as the "relation-ship"
    encoder")


    Args:
        language_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the language encoder.
        vision_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the visual encoder.
        pooled_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification, CLS, token) further processed
            by a Linear layer and a Tanh activation function. The Linear
        language_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        language_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        vision_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    tapas_output: Optional[torch.FloatTensor] = None
    vit_output: Optional[torch.FloatTensor] = None
    pooled_output: Optional[torch.FloatTensor] = None
    tapas_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vit_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    tapas_attentions: Optional[Tuple[torch.FloatTensor]] = None
    vit_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None




#Pretreined Model
class VisionTapasPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VisionTapasConfig
    base_model_prefix = "visiontapas"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

# Mostly copied from transformers.lxmert
class VisionTapasPooler(nn.Module):
    def __init__(self, config):
        super(VisionTapasPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



# VisionTapas Model
class VisionTapasModel(VisionTapasPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.lxmert_config = LxmertConfig(num_hidden_layers=self.config.x_layers)


        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k') # Load the pretrained checkpoint
        self.tapas = TapasModel.from_pretrained('google/tapas-base-finetuned-wtq') # Load the pretrained checkpoint
          
        # Cross Module Layers
        self.x_layers = nn.ModuleList([LxmertXLayer(self.lxmert_config) for _ in range(config.x_layers)])

        # Pooler
        self.pooler = VisionTapasPooler(self.lxmert_config)

        #Initialize weights
        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask, pixel_values, output_attentions=None, output_hidden_states=None, return_dict=None):

        tapas_inputs = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask,
                        'output_attentions': output_attentions, 'output_hidden_states': output_hidden_states}
        vit_inputs = {'pixel_values': pixel_values, 'output_attentions': output_attentions, 'output_hidden_states': output_hidden_states}

        tapas_outputs = self.tapas(**tapas_inputs)
        vit_outputs = self.vit(**vit_inputs)

        cross_encoder_attentions = () if output_attentions else None

        tapas_sequence = tapas_outputs[0]
        vit_sequence = vit_outputs[0]

        if output_hidden_states:
            tapas_hidden_states = tapas_outputs[2]
            vit_hidden_states = vit_outputs[2]
        if output_attentions:
            tapas_attentions = tapas_outputs[3]
            vit_attentions = vit_outputs[3]

        lang_feats = tapas_sequence
        visual_feats = vit_sequence
        lang_attention_mask = tapas_inputs['attention_mask']
        lang_attention_mask = torch.reshape(lang_attention_mask, (lang_attention_mask.size()[0], 1, 1, lang_attention_mask.size()[1])) #reshaping the mask to match the language features.
        visual_attention_mask = None # No Mask Since the ViT doesn't apply any masking or padding.

        # Run cross-modality layers
        for layer_module in self.x_layers:
            x_outputs = layer_module(
                lang_feats,
                lang_attention_mask,
                visual_feats,
                visual_attention_mask,
            )
            lang_feats, visual_feats = x_outputs[:2]
            if output_hidden_states:
                vit_hidden_states = vit_hidden_states + (visual_feats,)
                tapas_hidden_states = tapas_hidden_states + (lang_feats,)
            if cross_encoder_attentions is not None:
                cross_encoder_attentions = cross_encoder_attentions + (x_outputs[2],)


        pooled_output = self.pooler(lang_feats)
        all_hidden_states = ()
        if output_hidden_states:
            all_hidden_states = (tapas_hidden_states, vit_hidden_states)

        all_attentions = ()
        if output_attentions:
            all_attentions = (tapas_attentions, vit_attentions, cross_encoder_attentions)

        if not return_dict:
            return (pooled_output, lang_feats, visual_feats) + all_hidden_states + all_attentions


        return VisionTapasModelOutput(
            pooled_output=pooled_output,
            tapas_output=lang_feats,
            vit_output=visual_feats,
            tapas_hidden_states=tapas_hidden_states if output_hidden_states else None,
            vit_hidden_states=vit_hidden_states if output_hidden_states else None,
            tapas_attentions=tapas_attentions if output_attentions else None,
            vit_attentions=vit_attentions if output_attentions else None,
            cross_encoder_attentions=cross_encoder_attentions if output_attentions else None,
        )

