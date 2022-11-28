from utils.tapas_utils_new import parse_question, convert_logits_to_predictions, get_final_answer

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
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

from model.vision_tapas_for_classification import VisionTapasForClassification
from data.inference_dataset import VisionTapasInferenceDataset
from model.config import VisionTapasConfig
from args_parser import get_parserInference


def main():
    args = get_parserInference().parse_args()
    validation_folder = args.validation_folder
    qa_val_file = args.qa_val_file
    eval_batch_size = args.eval_batch_size
    out_dir = args.out_dir
    checkpoint_folder = args.checkpoint_folder

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    logger.info("Loading Tokenizer and Feat Extractor")
    # Tokenizer and feature extractor
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-base-finetuned-wtq')
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    logger.info("Loading Model")
    # Load from Checkpoint
    model = VisionTapasForClassification.from_pretrained(args.checkpoint_folder)

    #model.load_state_dict(torch.load(checkpoint_folder+"/pytorch_model.bin"))
    model = model.to("cuda")
    model.eval()
    
    val_dataset = VisionTapasInferenceDataset(args.qa_val_file, os.path.join(args.validation_folder, "tables"), os.path.join(args.validation_folder, "png"), tokenizer, feature_extractor)

    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
    logger.info("Prediction Starts")

    predicted_instances = [['imgname', 'query', 'predicted label']]
    for batch in tqdm(val_dataloader):
        
        tables = batch['table_name']
        questions = batch['question']
        del batch['table_name']
        del batch['question']

        input_batch = {}
        for key, val in batch.items():
            input_batch[key] = val.to('cuda')
        out = model(**input_batch)
        predicted_classes = out.cpu().detach().argmax(-1)

        for j in range(len(predicted_classes)):
            predicted_instances.append([tables[j], questions[j], predicted_classes[j]])

    predicted_df = pd.DataFrame(predicted_instances[1:], columns=predicted_instances[0])
    predicted_df.to_csv(out_dir+"predictions.csv", index=False)

if __name__ == '__main__':
    main()
