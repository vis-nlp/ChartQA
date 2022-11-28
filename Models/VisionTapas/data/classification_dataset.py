import os, json
import numpy as np
import torch.nn as nn
import torch
import pandas as pd
from PIL import Image

class VisionTapasForClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, qa_file_path, tables_folder, images_folder, tokenizer, feature_extractor):

        # Load QA file
        qa_file = open(qa_file_path, "r")
        self.instances = json.load(qa_file)

        self.tables_folder = tables_folder
        self.images_folder = images_folder
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        instance = self.instances[idx]
        image_index = str(instance["image_index"]).split(".")[0]
        question = instance["question"]
        answer = instance["answer"]

        # Load & Process table
        data_table = pd.read_csv(os.path.join(self.tables_folder, str(image_index)+".csv")).astype(str)
        encoding = self.tokenizer(table=data_table, queries=[question], padding="max_length", truncation=True, return_tensors="pt")

        # Load & Process Image
        image = Image.open(os.path.join(self.images_folder, str(image_index) + ".png")).convert("RGB")
        vis_inputs = self.feature_extractor(images=image, return_tensors="pt")

        # Remove Batch Dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        for key, val in vis_inputs.items():
            item[key] = val.squeeze(0)

        # Add labels
        item['labels'] = torch.tensor(answer) # answer is the index of the label in your one-hot encoding mapping. 
        return item

    def __len__(self):
        return len(self.instances)