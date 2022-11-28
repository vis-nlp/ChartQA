import argparse
import numpy as np
from typing import Dict
import os, pickle
import json
from random import shuffle
import time
import sys
import transformers
from transformers import ViTFeatureExtractor, TapasTokenizer, ViTModel, TapasModel
import torch.nn as nn
import torch
import pandas as pd
from transformers import EvalPrediction
from transformers import Trainer, TrainingArguments
import logging


from model.vision_tapas_for_question_answering import VisionTapasForQuestionAnswering
from data.question_answering_dataset import VisionTapasForQuestionAnsweringDataset
from model.config import VisionTapasConfig
from args_parser import get_parserQA


# Compute evaluation metrics to be displayed
def compute_metrics(p: EvalPrediction) -> Dict:
    #print("Dict:", p, '\n')
    # print('raw_predictions: ', p.predictions,  '\n')
    # for e in p.predictions:
    #     print(e.shape)
    print('labels: ', p.label_ids.shape, p.predictions.shape,'\n')
    preds = np.argmax(p.predictions, axis=-1)
    return {
        'accuracy': (preds == p.label_ids).mean()
    }

def main():

    # Parse command line arguments
    parser = get_parserQA()
    args = parser.parse_args()
    train_folder = args.train_folder
    validation_folder = args.validation_folder
    qa_train_file = args.qa_train_file
    qa_val_file = args.qa_val_file
    EPOCHS = args.EPOCHS
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    eval_num = args.eval_num
    eval_steps = args.eval_steps
    out_dir = args.out_dir
    num_labels = args.num_labels
    fixed_vocab_file = args.fixed_vocab_file

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load Classes Mapping File
    file = open(fixed_vocab_file, 'rb')
    fixed_vocab = pickle.load(file)
    # Add Main Operations
    aggregation_labels = {'0': 'NONE', '1': 'SUM', '2': 'AVERAGE', '3': 'COUNT', '4': 'Diff', '5': 'Ratio'}
    # Add Other Classes
    init_length = len(aggregation_labels)
    for i in range(len(fixed_vocab)):
        aggregation_labels[str(i + init_length)] = str(fixed_vocab[i])
    classes_mappings = dict((v, int(k)) for k, v in aggregation_labels.items())

    # Tokenizer and feature extractor
    tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    # Check if a checkpoint is provided
    if args.checkpoint_folder:
      model = VisionTapasForQuestionAnswering.from_pretrained(args.checkpoint_folder)
    else:
      # Initialize with the default parameters and config
      visiontapasconfig = VisionTapasConfig(x_layers=4)
      model = VisionTapasForQuestionAnswering(visiontapasconfig, aggregation_labels=aggregation_labels, args=args)
      # Load the Pretrained TaPas and ViT Models
      model.visiontapas.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k') # Load the pretrained checkpoint
      model.visiontapas.tapas = TapasModel.from_pretrained('google/tapas-base-finetuned-wtq') # Load the pretrained checkpoint
      # Edit TaPas Configs
      model.visiontapas.tapas.config.answer_loss_cutoff = args.answer_loss_cutoff #50
      model.visiontapas.tapas.config.select_one_column = args.select_one_column #True
      model.visiontapas.tapas.config.cell_selection_preference = args.cell_selection_preference #0.001
      model.visiontapas.tapas.config.num_aggregation_labels = len(aggregation_labels)
      model.visiontapas.tapas.config.aggregation_labels = aggregation_labels 

    train_dataset = VisionTapasForQuestionAnsweringDataset(args.qa_train_file, os.path.join(args.train_folder, "tables"), os.path.join(args.train_folder, "png"), tokenizer, feature_extractor, classes_mappings=classes_mappings)
    val_dataset = VisionTapasForQuestionAnsweringDataset(args.qa_val_file, os.path.join(args.validation_folder, "tables"), os.path.join(args.validation_folder, "png"), tokenizer, feature_extractor, classes_mappings=classes_mappings)


    training_args = TrainingArguments(
        output_dir=args.out_dir,  # output directory
        logging_dir=args.out_dir,  # directory for storing logs
        num_train_epochs=args.EPOCHS,  # total number of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.eval_batch_size,  # batch size for evaluation
        warmup_ratio= args.warmup_ratio, #0.1,  # number of warmup steps for learning rate scheduler
        learning_rate= args.learning_rate,
        #fp16=True,
        weight_decay= args.weight_decay, #0.01  # strength of weight decay
        dataloader_num_workers=args.num_workers,

        save_strategy="steps",
        evaluation_strategy = "steps",
        logging_steps = eval_steps,
        eval_steps = eval_steps,
        save_steps = eval_steps,
        load_best_model_at_end=True,
        #metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,  # the instantiated  Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        #compute_metrics = compute_metrics # metrics function
    )
    trainer.train()
    trainer.model.save_pretrained(os.path.join(out_dir, "best_model"))

# Run the training
if __name__ == '__main__':
    main()