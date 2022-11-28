import argparse
import numpy as np
from typing import Dict
import os
import json
from random import shuffle
import time
import sys
import transformers
from transformers import ViTFeatureExtractor, TapasTokenizer
import torch.nn as nn
import torch
import pandas as pd
from transformers import EvalPrediction
from transformers import Trainer, TrainingArguments, ViTModel, TapasModel
import logging


from model.vision_tapas_for_classification import VisionTapasForClassification
from data.classification_dataset import VisionTapasForClassificationDataset
from model.config import VisionTapasConfig
from args_parser import get_parser


# Compute evaluation metrics to be displayed
def compute_metrics(p: EvalPrediction) -> Dict:
    #print("Dict:", p, '\n')
    # print('raw_predictions: ', p.predictions,  '\n')
    # for e in p.predictions:
    #     print(e.shape)
    #print('labels: ', p.label_ids.shape, p.predictions.shape,'\n')
    preds = np.argmax(p.predictions, axis=-1)
    return {
        'accuracy': (preds == p.label_ids).mean()
    }

def main():

    # Parse command line arguments 
    parser = get_parser()
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
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
  
    logger.info("Loading Tokenizer and Feat Extractor")
    # Tokenizer and feature extractor
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-base-finetuned-wtq')
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    logger.info("Loading Model")
    # Check if a checkpoint is provided
    if args.checkpoint_folder:
        model = VisionTapasForClassification.from_pretrained(args.checkpoint_folder)
    else:
        # Initialize with the default parameters and config
        visiontapasconfig = VisionTapasConfig(x_layers=4, num_labels=args.num_labels)
        model = VisionTapasForClassification(visiontapasconfig)
        # Load the Pretrained TaPas and ViT Models
        model.visiontapas.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k') # Load the pretrained checkpoint
        model.visiontapas.tapas = TapasModel.from_pretrained('google/tapas-base-finetuned-wtq') # Load the pretrained checkpoint

    logger.info("load dataset")
    train_dataset = VisionTapasForClassificationDataset(args.qa_train_file, os.path.join(args.train_folder, "tables"), os.path.join(args.train_folder, "png"), tokenizer, feature_extractor)
    val_dataset = VisionTapasForClassificationDataset(args.qa_val_file, os.path.join(args.validation_folder, "tables"), os.path.join(args.validation_folder, "png"), tokenizer, feature_extractor)

    logger.info("Training Arguments")
    training_args = TrainingArguments(
        output_dir=args.out_dir,  # output directory
        logging_dir=args.out_dir,  # directory for storing logs
        num_train_epochs=args.EPOCHS,  # total number of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.eval_batch_size,  # batch size for evaluation
        warmup_ratio= args.warmup_ratio, #0.1,  # number of warmup steps for learning rate scheduler
        learning_rate= args.learning_rate,
        fp16=True,
        weight_decay= args.weight_decay, #0.01  # strength of weight decay
        dataloader_num_workers=args.num_workers,

        save_strategy="steps",
        evaluation_strategy = "steps",
        logging_steps = eval_steps,
        eval_steps = eval_steps,
        save_steps = eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,  # the instantiated  Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        compute_metrics = compute_metrics # metrics function
    )
    logger.info("Start Training")
    trainer.train()
    trainer.model.save_pretrained(os.path.join(out_dir, "best_model"))
    # output = trainer.predict(val_dataset)
    # predicted = np.argmax(output.predictions, axis=-1)
    # data_frame = pd.DataFrame(val_dataset.instances)
    # data_frame['Predicted Answer'] = predicted
    # data_frame.to_csv("/home/masry20/projects/def-enamul/masry20/ChartQA/predictions/dvqa_val_easy_partial_1.csv", index=False)

# Run the training
if __name__ == '__main__':
    main()
