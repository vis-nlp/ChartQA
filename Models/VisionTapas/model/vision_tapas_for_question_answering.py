
import logging
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
from.vision_tapas import VisionTapasPreTrainedModel, VisionTapasModel
from .tapas_utils import IndexMap, ProductIndexMap, reduce_mean,compute_token_logits, compute_column_logits, _calculate_aggregate_mask, gather
from .tapas_utils import _single_column_cell_selection_loss, _calculate_aggregation_loss, _calculate_regression_loss

EPSILON_ZERO_DIVISION = 1e-10
CLOSE_ENOUGH_TO_LOG_ZERO = -10000.0

try:
    from torch_scatter import scatter
except OSError:
    logger.error(
        "TAPAS models are not usable since `torch_scatter` can't be loaded."
        "It seems you have `torch_scatter` installed with the wrong CUDA version."
        "Please try to reinstall it following the instructions here: https://github.com/rusty1s/pytorch_scatter."
    )

class VisionTapasForQuestionAnsweringOutput(ModelOutput):
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
    question_answering_score: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[torch.FloatTensor] = None
    aggregation_logits: Optional[torch.FloatTensor] = None



class VisionTapasForQuestionAnswering(VisionTapasPreTrainedModel):
    def __init__(self, config, aggregation_labels, args):
        super().__init__(config)
        # Configuration
        self.config = config
        self.lxmert_config = LxmertConfig(num_hidden_layers=config.x_layers)
        self.num_labels = config.num_labels

        # VisionTapas backbone
        self.visiontapas = VisionTapasModel(config)

        # Edit TaPas Configs
        self.visiontapas.tapas.config.answer_loss_cutoff = args.answer_loss_cutoff #50
        self.visiontapas.tapas.config.select_one_column = args.select_one_column #True
        self.visiontapas.tapas.config.cell_selection_preference = args.cell_selection_preference #0.001
        self.visiontapas.tapas.config.num_aggregation_labels = len(aggregation_labels)
        self.visiontapas.tapas.config.aggregation_labels = aggregation_labels 
        
        

        # dropout (only used when training)
        self.dropout = nn.Dropout(self.visiontapas.tapas.config.hidden_dropout_prob)

        # cell selection heads
        if self.visiontapas.tapas.config.init_cell_selection_weights_to_zero:
            # init_cell_selection_weights_to_zero: Whether the initial weights should be
            # set to 0. This ensures that all tokens have the same prior probability.
            self.output_weights = nn.Parameter(torch.zeros(self.visiontapas.tapas.config.hidden_size))
            self.column_output_weights = nn.Parameter(torch.zeros(self.visiontapas.tapas.config.hidden_size))
        else:
            self.output_weights = nn.Parameter(torch.empty(self.visiontapas.tapas.config.hidden_size))
            nn.init.normal_(
                self.output_weights, std=self.visiontapas.tapas.config.initializer_range
            )  # here, a truncated normal is used in the original implementation
            self.column_output_weights = nn.Parameter(torch.empty(self.visiontapas.tapas.config.hidden_size))
            nn.init.normal_(
                self.column_output_weights, std=self.visiontapas.tapas.config.initializer_range
            )  # here, a truncated normal is used in the original implementation
        self.output_bias = nn.Parameter(torch.zeros([]))
        self.column_output_bias = nn.Parameter(torch.zeros([]))

        # aggregation head
        if self.visiontapas.tapas.config.num_aggregation_labels > 0:
            self.aggregation_classifier = nn.Linear(self.visiontapas.tapas.config.hidden_size, self.visiontapas.tapas.config.num_aggregation_labels)

        # Loss function
        self.loss = nn.CrossEntropyLoss()

        # Initialize weights
        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask, pixel_values=None, inputs_embeds=None,
                labels=None, output_attentions=None, output_hidden_states=None, return_dict=None, table_mask = None, aggregation_labels = None,
                float_answer = None, numeric_values = None,numeric_values_scale = None, class_labels=None, class_labels_mask=None):
        r"""
        labels: (``Torch.Tensor`` of shape ``(batch_size)``, `optional`):
            A one-hot representation of the correct answer
        Returns:
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
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
        sequence_output = visiontapas_output.tapas_output
        outputs = (pooled_output, sequence_output)

        sequence_output = self.dropout(sequence_output)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise("Input ids can't be None")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Construct indices for the table.
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                (*input_shape, len(self.visiontapas.tapas.config.type_vocab_sizes)), dtype=torch.long, device=device
            )

        token_types = [
            "segment_ids",
            "column_ids",
            "row_ids",
            "prev_labels",
            "column_ranks",
            "inv_column_ranks",
            "numeric_relations",
        ]

        row_ids = token_type_ids[:, :, token_types.index("row_ids")]
        column_ids = token_type_ids[:, :, token_types.index("column_ids")]

        row_index = IndexMap(
            indices=torch.min(row_ids, torch.as_tensor(self.visiontapas.tapas.config.max_num_rows - 1, device=row_ids.device)),
            num_segments=self.visiontapas.tapas.config.max_num_rows,
            batch_dims=1,
        )
        col_index = IndexMap(
            indices=torch.min(column_ids, torch.as_tensor(self.visiontapas.tapas.config.max_num_columns - 1, device=column_ids.device)),
            num_segments=self.visiontapas.tapas.config.max_num_columns,
            batch_dims=1,
        )
        cell_index = ProductIndexMap(row_index, col_index)

        # Masks.
        input_shape = input_ids.size() if input_ids is not None else inputs_embeds.size()[:-1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        # Table cells only, without question tokens and table headers.
        if table_mask is None:
            table_mask = torch.where(row_ids > 0, torch.ones_like(row_ids), torch.zeros_like(row_ids))
        # torch.FloatTensor[batch_size, seq_length]
        input_mask_float = attention_mask.float().to(device)
        table_mask_float = table_mask.float().to(device)
        # Mask for cells that exist in the table (i.e. that are not padding).
        cell_mask, _ = reduce_mean(input_mask_float, cell_index)

        # Compute logits per token. These are used to select individual cells.
        logits = compute_token_logits(sequence_output, self.visiontapas.tapas.config.temperature, self.output_weights, self.output_bias)
        # Compute logits per column. These are used to select a column.
        column_logits = None
        if self.visiontapas.tapas.config.select_one_column:
            column_logits = compute_column_logits(
                sequence_output,
                self.column_output_weights,
                self.column_output_bias,
                cell_index,
                cell_mask,
                self.visiontapas.tapas.config.allow_empty_column_selection,
            )
        #print("column logits", column_logits)
        # Aggregation logits
        logits_aggregation = None
        if self.visiontapas.tapas.config.num_aggregation_labels > 0:
            logits_aggregation = self.aggregation_classifier(pooled_output)
        # Total loss calculation
        total_loss = 0.0
        calculate_loss = False
        if labels is not None:
            calculate_loss = True
            is_supervised = not self.visiontapas.tapas.config.num_aggregation_labels > 0 or not self.visiontapas.tapas.config.use_answer_as_supervision
            # Semi-supervised cell selection in case of no aggregation:
            # If the answer (the denotation) appears directly in the table we might
            # select the answer without applying any aggregation function. There are
            # some ambiguous cases, see utils._calculate_aggregate_mask for more info.
            # `aggregate_mask` is 1 for examples where we chose to aggregate and 0
            #  for examples where we chose to select the answer directly.
            # `labels` encodes the positions of the answer appearing in the table.
            if is_supervised:
                aggregate_mask = None
            else:
                if float_answer is not None:
                    assert (
                        labels.shape[0] == float_answer.shape[0]
                    ), "Make sure the answers are a FloatTensor of shape (batch_size,)"
                    # <float32>[batch_size]
                    #print(float_answer, self.visiontapas.tapas.config.cell_selection_preference)
                    aggregate_mask = _calculate_aggregate_mask(
                        float_answer,
                        pooled_output,
                        self.visiontapas.tapas.config.cell_selection_preference,
                        labels,
                        self.aggregation_classifier,
                    )
                else:
                    raise ValueError("You have to specify float answers in order to calculate the aggregate mask")

            # Cell selection log-likelihood
            if self.visiontapas.tapas.config.average_logits_per_cell:
                logits_per_cell, _ = reduce_mean(logits, cell_index)
                logits = gather(logits_per_cell, cell_index)
            #dist_per_token = torch.distributions.Bernoulli(logits=logits) # Moved to the if condition below to avoid the NanLogits in single column cases

            # Compute cell selection loss per example.
            selection_loss_per_example = None
            if not self.visiontapas.tapas.config.select_one_column:
                dist_per_token = torch.distributions.Bernoulli(logits=logits)
                weight = torch.where(
                    labels == 0,
                    torch.ones_like(labels, dtype=torch.float32),
                    self.visiontapas.tapas.config.positive_label_weight * torch.ones_like(labels, dtype=torch.float32),
                )
                float_labels = labels.type('torch.FloatTensor').to(device)
                selection_loss_per_token = -dist_per_token.log_prob(float_labels) * weight
                selection_loss_per_example = torch.sum(selection_loss_per_token * input_mask_float, dim=1) / (
                    torch.sum(input_mask_float, dim=1) + EPSILON_ZERO_DIVISION
                )
            else:
                selection_loss_per_example, logits = _single_column_cell_selection_loss(
                    logits, column_logits, labels, cell_index, col_index, cell_mask
                )
                dist_per_token = torch.distributions.Bernoulli(logits=logits)

            # Supervised cell selection
            if self.visiontapas.tapas.config.disable_per_token_loss:
                pass
            elif is_supervised:
                total_loss += torch.mean(selection_loss_per_example)
            else:
                # For the not supervised case, do not assign loss for cell selection
                total_loss += torch.mean(selection_loss_per_example * (1.0 - aggregate_mask) * (1.0 - class_labels_mask))
            # Semi-supervised regression loss and supervised loss for aggregations
            if self.visiontapas.tapas.config.num_aggregation_labels > 0:
                if is_supervised:
                    # Note that `aggregate_mask` is None if the setting is supervised.
                    if aggregation_labels is not None:
                        assert (
                            labels.shape[0] == aggregation_labels.shape[0]
                        ), "Make sure the aggregation labels are a LongTensor of shape (batch_size,)"
                        per_example_additional_loss = _calculate_aggregation_loss(
                            logits_aggregation,
                            aggregate_mask,
                            aggregation_labels,
                            self.visiontapas.tapas.config.use_answer_as_supervision,
                            self.visiontapas.tapas.config.num_aggregation_labels,
                            self.visiontapas.tapas.config.aggregation_loss_weight,
                            class_labels_mask,
                            class_labels
                        )
                    else:
                        raise ValueError(
                            "You have to specify aggregation labels in order to calculate the aggregation loss"
                        )
                else:
                    # Set aggregation labels to zeros
                    aggregation_labels = torch.zeros(labels.shape[0], dtype=torch.long, device=labels.device)
                    per_example_additional_loss = _calculate_aggregation_loss(
                        logits_aggregation,
                        aggregate_mask,
                        aggregation_labels,
                        self.visiontapas.tapas.config.use_answer_as_supervision,
                        self.visiontapas.tapas.config.num_aggregation_labels,
                        self.visiontapas.tapas.config.aggregation_loss_weight,
                        class_labels_mask,
                        class_labels
                    )

                if self.visiontapas.tapas.config.use_answer_as_supervision:
                    if numeric_values is not None and numeric_values_scale is not None:
                        assert numeric_values.shape == numeric_values_scale.shape
                        # Add regression loss for numeric answers which require aggregation.
                        answer_loss, large_answer_loss_mask = _calculate_regression_loss(
                            float_answer,
                            aggregate_mask,
                            dist_per_token,
                            numeric_values,
                            numeric_values_scale,
                            table_mask_float,
                            logits_aggregation,
                            self.visiontapas.tapas.config,
                        )
                        per_example_additional_loss += answer_loss
                        # Zero loss for examples with answer_loss > cutoff.
                        per_example_additional_loss *= large_answer_loss_mask
                        #per_example_additional_loss *= (1-class_labels_mask)
                    else:
                        raise ValueError(
                            "You have to specify numeric values and numeric values scale in order to calculate the regression loss"
                        )

                total_loss += torch.mean(per_example_additional_loss)

        else:
            # if no label ids are provided, set them to zeros in order to properly compute logits
            if self.visiontapas.tapas.config.select_one_column:
                labels = torch.zeros_like(logits)
                _, logits = _single_column_cell_selection_loss(
                    logits, column_logits, labels, cell_index, col_index, cell_mask
                )

        if not return_dict:
            output = (logits, logits_aggregation) + outputs[2:]
            return ((total_loss,) + output) if calculate_loss else output


        return VisionTapasForQuestionAnsweringOutput(
            loss=total_loss,
            logits=logits,
            aggregation_logits=logits_aggregation,
        )
