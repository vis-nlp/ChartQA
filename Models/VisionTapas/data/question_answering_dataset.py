import os, json
import numpy as np
import torch.nn as nn
import torch
import pandas as pd
from PIL import Image
from .tapas_utils import parse_question

class VisionTapasForQuestionAnsweringDataset(torch.utils.data.Dataset):
    def __init__(self, qa_file_path, tables_folder, images_folder, tokenizer, feature_extractor, classes_mappings):

        # Load QA file
        qa_file = open(qa_file_path, "r")
        self.instances = json.load(qa_file)

        self.tables_folder = tables_folder
        self.images_folder = images_folder
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.classes_mappings = classes_mappings

    def __getitem__(self, idx):
        instance = self.instances[idx]
        image_index = str(instance["image_index"]).split(".")[0]
        question = instance["question"]
        answer = instance["answer"]

        # Load table
        data_table = pd.read_csv(os.path.join(self.tables_folder, str(image_index)+".csv"), encoding='utf8', index_col=False, header=None).astype(str)
        data_table.columns = data_table.iloc[0, :].astype(str)
        data_table = data_table.applymap(lambda x: str(x).strip('%')) # Clean the data table from all the % signs to treat them as numbers
        # Process Answer
        answer_type, answer_list = answer
        if answer_type == 'FIXED/OPEN':
            if len(answer_list) == 1 and answer_list[0].lower() in self.classes_mappings:
                class_labels = self.classes_mappings[answer_list[0].lower()]
                class_labels_mask = 1
                answer_texts = [str(data_table.iloc[0, 0])] # dummy answer which will be ignored by the model in loss computation by using the class_labels_mask. 
            else:
                class_labels = 0
                class_labels_mask = 0
                answer_texts = [str(x) for x in answer_list]
        elif answer_type in ['Ratio', 'Diff']:
            class_labels = self.classes_mappings[answer_type]
            class_labels_mask = 0
            answer_texts = [str(x) for x in answer_list]


        
        # Parse the question and get the answer coordinates. 
        question, answer_texts, answer_coordinates, float_value, aggregation_function = parse_question(table=data_table, question=[question], answer_texts=answer_texts)

        if answer_coordinates is None:
            answer_coordinates = [(-1, -1)]
        inputs = self.tokenizer(table=data_table, queries=question, answer_text=answer_texts,
                                answer_coordinates=[answer_coordinates], float_value=float_value, padding="max_length",
                                return_tensors="pt", truncation =True)

        # Load & Process Image
        image = Image.open(os.path.join(self.images_folder, str(image_index) + ".png")).convert("RGB")
        vis_inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs['pixel_values'] = vis_inputs['pixel_values']
        
        # Add Float Value fo the inputs.
        if float_value is not None:
            inputs['float_answer'] = torch.tensor([float_value])
        else:
            inputs['float_answer'] = torch.tensor([float('nan')])

        # Add Class labels and Class Labels Mask
        inputs['class_labels'] = torch.LongTensor([class_labels])
        inputs['class_labels_mask'] = torch.LongTensor([class_labels_mask])
        
        item = {key: val.squeeze(0) for key, val in inputs.items()}


        return item

    def __len__(self):
        return len(self.instances)