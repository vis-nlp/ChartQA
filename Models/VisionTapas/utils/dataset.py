import os
import numpy as np
import torch.nn as nn
import torch
import pandas as pd
from PIL import Image
from .tapas_utils_new import parse_question
import ast 

class VisionTapasDataset(torch.utils.data.Dataset):
    def __init__(self, instances, tables, images_folder, tokenizer, feature_extractor):
        # Qa pairs.
        self.instances = instances
        self.tables = tables
        self.images_folder = images_folder
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        instance = self.instances[idx]
        image_index = instance["image_index"].split(".")[0]
        question = instance["question"]
        answer = instance["answer"]

        # Process table
        df = self.tables[str(image_index)]
        new_df = df.copy()#.iloc[:, 1:].copy()
        new_df = new_df.astype(str)

        encoding = self.tokenizer(table=new_df, queries=[question], padding="max_length", truncation=True, return_tensors="pt")

        image = Image.open(self.images_folder + str(image_index) + ".png").convert("RGB")
        vis_inputs = self.feature_extractor(images=image, return_tensors="pt")

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        for key, val in vis_inputs.items():
            item[key] = val.squeeze(0)
        item['labels'] = torch.tensor(answer)
        return item

    def __len__(self):
        return len(self.instances)

class TapasForSequenceClassificationVisionDataset(torch.utils.data.Dataset):
    def __init__(self, instances, tables_folder, images_folder, tokenizer, feature_extractor):
        # Qa pairs.
        self.instances = instances
        self.tables_folder = tables_folder
        self.tokenizer = tokenizer
        self.images_folder = images_folder
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        instance = self.instances.iloc[idx]
        image_name = instance["imgname"]
        question = instance["query"]
        answer = instance["label"]

        df = pd.read_csv(self.tables_folder+image_name+".csv")

        # Process table
        new_df = df.copy()#.iloc[:, 1:].copy()
        new_df = new_df.astype(str)
        encoding = self.tokenizer(table=new_df, queries=[question], padding="max_length", truncation=True, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        #image = Image.open(self.images_folder + str(image_name) + ".png").convert("RGB")
        #vis_inputs = self.feature_extractor(images=image, return_tensors="pt")
        #for key, val in vis_inputs.items():
        #    item[key] = val.squeeze(0)

        item['labels'] = torch.tensor(0 if answer == "No" else 1)
        return item

    def __len__(self):
        return len(self.instances)
class TapasForSequenceClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, instances, tables_folder, tokenizer):
        # Qa pairs.
        self.instances = instances
        self.tables_folder = tables_folder
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        instance = self.instances.iloc[idx]
        image_name = instance["imgname"]
        question = instance["query"]
        answer = instance["label"]

        df = pd.read_csv(self.tables_folder+image_name+".csv")

        # Process table
        new_df = df.copy()#.iloc[:, 1:].copy()
        new_df = new_df.astype(str)
        encoding = self.tokenizer(table=new_df, queries=[question], padding="max_length", truncation=True, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(0 if answer == "No" else 1)
        return item

    def __len__(self):
        return len(self.instances)

class TapasDataset(torch.utils.data.Dataset):
    def __init__(self, instances, tables_folder, tokenizer):
        # Qa pairs.
        self.instances = instances
        self.tables_folder = tables_folder
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        instance = self.instances.iloc[idx]
        image_name = instance["imgname"]
        query = instance["query"]
        answer = instance["label"]

        df = pd.read_csv(self.tables_folder+image_name+".csv")
        new_df = df.copy()  # .iloc[:, 1:].copy()
        new_df = new_df.astype(str)

        question, answer_texts, answer_coordinates, float_value, aggregation_function = parse_question(table=new_df, question=[query], answer_texts=[str(answer)])  # , float_value=str(answer))
        if answer_coordinates is None:
            answer_coordinates = [(-1, -1)]
        inputs = self.tokenizer(table=new_df, queries=question, answer_text=answer_texts,
                           answer_coordinates=[answer_coordinates], float_value=float_value, padding="max_length",
                           return_tensors="pt")

        if float_value is not None:
            inputs['float_answer'] = torch.tensor([float_value])
        else:
            inputs['float_answer'] = torch.tensor([float('nan')])


        item = {key: val.squeeze(0) for key, val in inputs.items()}
        return item

    def __len__(self):
        return len(self.instances)


def process_str_list(text):
    text = text[1:-1]
    return [x.strip().replace('\'', '') for x in text.split(",")]
def check_answer_lst(text):
    if "[" in str(text) and "]" in str(text):
        return True
    return False
class VisionTapasCombinedDataset(torch.utils.data.Dataset):
    def __init__(self, instances, tables, images_folder, tokenizer, feature_extractor, answers_to_indices):
        # Qa pairs.
        self.instances = instances
        self.tables = tables
        self.images_folder = images_folder
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.answers_to_indices = answers_to_indices
        self.supervised_ops = {'Diff': 4, 'Ratio': 5}

    def __getitem__(self, idx):
        instance = self.instances.iloc[idx]
        image_name = str(instance["imgname"]).split(".")[0]
        query = instance["query"]
        answer = str(instance["label"])
        orig_answer = answer
        # Process table
        try:
            df = self.tables[str(image_name)]
        except:
        # Return random item
            while True:
                try:
                    return self.__getitem__(np.random.randint(0, len(self.instances)-1))
                except:
                    continue

        new_df = df.copy()  # .iloc[:, 1:].copy()
        new_df = new_df.astype(str)

        #print("Image name:", image_name)
        #print(query)
        # cHECK IF ANSWER IS LIST
        answer_is_lst = check_answer_lst(answer)


        if answer.lower() in self.answers_to_indices:
            class_labels = self.answers_to_indices[answer.lower()]
            class_labels_mask = 1
            answer = str(new_df.iloc[0, 0])
        elif answer_is_lst:
            if len(process_str_list(answer))>=3 and process_str_list(answer)[2] in self.supervised_ops:
                class_labels = int(self.supervised_ops[ast.literal_eval(answer)[2]])
                #print("class_labels", class_labels)
                class_labels_mask = 0  # Special cases
                answer = str(ast.literal_eval(answer)[:2])  # First two indices
            else:
                class_labels_mask = 0
                class_labels = 0
        else:
            class_labels = 0
            class_labels_mask = 0

        if answer_is_lst:
            answer_1 = process_str_list(answer)
            answer_texts = [str(elt) for elt in answer_1]
        else:
            answer_texts = [str(answer)]
        try:
            question, answer_texts, answer_coordinates, float_value, aggregation_function = parse_question(table=new_df, question=[query], answer_texts=answer_texts)  # , float_value=str(answer))
        except:
            #Return random item
            while True:
                try:
                    return self.__getitem__(np.random.randint(0, len(self.instances)-1))
                except:
                    continue

        if answer_coordinates is None:
            answer_coordinates = [(-1, -1)]
        inputs = self.tokenizer(table=new_df, queries=question, answer_text=answer_texts,
                                answer_coordinates=[answer_coordinates], float_value=float_value, padding="max_length",
                                return_tensors="pt", truncation =True)
        image = Image.open(self.images_folder + str(image_name) + ".png").convert("RGB")
        vis_inputs = self.feature_extractor(images=image, return_tensors="pt")

        # inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        for key, val in vis_inputs.items():
            inputs[key] = val.squeeze(0)
        if float_value is not None:
            inputs['float_answer'] = torch.tensor([float_value])
        else:
            inputs['float_answer'] = torch.tensor([float('nan')])

        #print("class_labels batch", class_labels)
        inputs['class_labels'] = torch.LongTensor([class_labels])
        inputs['class_labels_mask'] = torch.LongTensor([class_labels_mask])
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        # item['answer'] = answer_texts
        #item['question'] = question
        # item['table'] = [str(image_name)]
        # item['orig answer'] = orig_answer
        return item

    def __len__(self):
        return len(self.instances)

class VisionTapasCombinedDatasetPlotQA(torch.utils.data.Dataset):
    def __init__(self, instances, tables, images_folder, tokenizer, feature_extractor, answers_to_indices):
        # Qa pairs.
        self.instances = instances
        self.tables = tables
        self.images_folder = images_folder
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.answers_to_indices = answers_to_indices
        self.supervised_ops = {'Diff': 4, 'Ratio': 5}

    def __getitem__(self, idx):
        instance = self.instances.iloc[idx]
        image_name = instance["imgname"]
        query = instance["query"]
        answer = instance["label"]
        orig_answer = answer
        # Process table
        df = self.tables[str(image_name)]
        new_df = df.copy()  # .iloc[:, 1:].copy()
        new_df = new_df.astype(str)

        #print("Image name:", image_name)
        #print(query)
        # cHECK IF ANSWER IS LIST
        try:
            answer_is_lst = isinstance(ast.literal_eval(answer), list)
        except:
            answer_is_lst = False

        if answer in self.answers_to_indices:
            class_labels = self.answers_to_indices[answer]
            class_labels_mask = 1
            answer = str(new_df.iloc[0, 0])
        elif answer_is_lst and ast.literal_eval(answer)[2] in self.supervised_ops:
            class_labels = int(self.supervised_ops[ast.literal_eval(answer)[2]])
            #print("class_labels", class_labels)
            class_labels_mask = 0  # Special cases
            answer = str(ast.literal_eval(answer)[:2])  # First two indices

        else:
            class_labels = 0
            class_labels_mask = 0

        try:
            answer_1 = ast.literal_eval(answer)
            answer_texts = [str(elt) for elt in answer_1]
        except:
            answer_texts = [str(answer)]
        #print("Answer texts", answer_texts)
        try:
            question, answer_texts, answer_coordinates, float_value, aggregation_function = parse_question(table=new_df, question=[query], answer_texts=answer_texts)  # , float_value=str(answer))
        except:
            # Return random item
            while True:
                try:
                    return self.__getitem__(np.random.randint(0, len(self.instances)-1))
                except:
                    continue

        if answer_coordinates is None:
            answer_coordinates = [(-1, -1)]
        # print(question, answer_texts, answer_coordinates, float_value)
        inputs = self.tokenizer(table=new_df, queries=question, answer_text=answer_texts,
                                answer_coordinates=[answer_coordinates], float_value=float_value, padding="max_length",
                                return_tensors="pt")
        image = Image.open(self.images_folder + str(image_name) + ".png").convert("RGB")
        vis_inputs = self.feature_extractor(images=image, return_tensors="pt")

        # inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        for key, val in vis_inputs.items():
            inputs[key] = val.squeeze(0)
        if float_value is not None:
            inputs['float_answer'] = torch.tensor([float_value])
        else:
            inputs['float_answer'] = torch.tensor([float('nan')])

        #print("class_labels batch", class_labels)
        inputs['class_labels'] = torch.LongTensor([class_labels])
        inputs['class_labels_mask'] = torch.LongTensor([class_labels_mask])
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        # item['answer'] = answer_texts
        # item['question'] = question
        # item['table'] = [str(image_name)]
        # item['orig answer'] = orig_answer
        return item

    def __len__(self):
        return len(self.instances)

class VisionTapasCombinedDatasetValidation(torch.utils.data.Dataset):
    def __init__(self, instances, tables, images_folder, tokenizer, feature_extractor, answers_to_indices):
        # Qa pairs.
        self.instances = instances
        self.tables = tables
        self.images_folder = images_folder
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.answers_to_indices = answers_to_indices
        self.supervised_ops = {'Diff': 4, 'Ratio': 5}

    def __getitem__(self, idx):
        instance = self.instances.iloc[idx]
        image_name = instance["imgname"].split(".")[0]
        query = instance["query"]
        answer = instance["label"]
        orig_answer = answer
        # Process table
        df = self.tables[str(image_name)]
        new_df = df.copy()  # .iloc[:, 1:].copy()
        new_df = new_df.astype(str)

        #print("Image name:", image_name)
        #print(query)

        # print(question, answer_texts, answer_coordinates, float_value)
        inputs = self.tokenizer(table=new_df, queries=[query], padding="max_length", return_tensors="pt")
        image = Image.open(self.images_folder + str(image_name) + ".png").convert("RGB")
        vis_inputs = self.feature_extractor(images=image, return_tensors="pt")

        # inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        for key, val in vis_inputs.items():
            inputs[key] = val.squeeze(0)

        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['question'] = query
        item['table'] = str(image_name)
        item['orig answer'] = orig_answer
        return item

    def __len__(self):
        return len(self.instances)

class T5Dataset(torch.utils.data.Dataset):
    def __init__(self, instances, tokenizer):
        # Qa pairs.
        self.instances = instances
        self.inputs = instances["Input"].values
        self.outputs = instances["Output"].values
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        input = self.inputs[idx]
        output = self.outputs[idx]
        inputs = self.tokenizer(str(input), padding="max_length", truncation=True, return_tensors='pt')
        labels = self.tokenizer(str(output), padding="max_length", truncation=True, return_tensors='pt').input_ids

        inputs['labels'] = labels
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)

        return inputs

    def __len__(self):
        return len(self.inputs)

class T5BboxesDataset(torch.utils.data.Dataset):
    def __init__(self, instances, tokenizer):
        # Qa pairs.
        self.instances = instances
        self.inputs = instances["Input"].values
        self.outputs = instances["Output"].values
        self.bboxes = instances['bboxes_text'].values
        #print(self.bboxes[0])
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        input = self.inputs[idx]
        output = self.outputs[idx]
        bboxes_text = self.bboxes[idx]
        bboxes_pre = str(bboxes_text).split("|")
        bboxes = []
        for bbox_pre in bboxes_pre:
            if bbox_pre == '':
                continue
            bbox = [float(x) for x in bbox_pre.split(",")]
            if len(bbox) != 4:
                continue
            bboxes.append(bbox)
        bboxes = np.array(bboxes)


        inputs = self.tokenizer(str(input), padding="max_length", truncation=True, return_tensors='pt')
        input_ids = inputs.input_ids
        # Prepare the bounding boxes tensor.
        seq_length = 512
        sep_index = (input_ids[0] == 1).nonzero(as_tuple=True)[0].tolist()[0]
        sep_indices = [sep_index] + (input_ids[0] == 1820).nonzero(as_tuple=True)[0].tolist() # 1820 for |

        padd_bbox = np.array([0, 0, 0, 0])
        bboxes_input_array = []
        #Padding for title
        bboxes_input_array.append(np.repeat(padd_bbox[np.newaxis, :], sep_index+1, axis=0))
        #BBoxes for text tokens
        for i in range(0, len(sep_indices) - 1):
            st_idx = sep_indices[i]
            end_idx = sep_indices[i + 1]
            bboxes_input_array.append(np.repeat(bboxes[i][np.newaxis, :], end_idx - st_idx, axis=0))
        #bboxes_input_array.append(np.repeat(padd_bbox[np.newaxis, :], seq_length - sep_indices[-1] - 1, axis=0))
        bboxes_num = sum([len(x) for x in bboxes_input_array])
        while bboxes_num < seq_length:
            bboxes_input_array.append(padd_bbox[np.newaxis, :])
            bboxes_num = sum([len(x) for x in bboxes_input_array])
        bboxes_input_tensor = torch.FloatTensor(np.concatenate(bboxes_input_array, axis=0)).unsqueeze(0)
        inputs['bboxes'] = bboxes_input_tensor
        ###

        labels = self.tokenizer(str(output), padding="max_length", truncation=True, return_tensors='pt').input_ids
        inputs['labels'] = labels
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)

        return inputs

    def __len__(self):
        return len(self.inputs)
class VisionT5Dataset(torch.utils.data.Dataset):
    def __init__(self, instances, tokenizer, feature_extractor, images_folder):
        # Qa pairs.
        self.instances = instances
        self.inputs = instances["Input"].values
        self.outputs = instances["Output"].values
        self.images_indices = instances["Image Index"].values
        self.tokenizer = tokenizer

        self.images_folder = images_folder
        self.feature_extractor = feature_extractor


    def __getitem__(self, idx):
        input = self.inputs[idx]
        output = self.outputs[idx]
        image_index = self.images_indices[idx]

        # Image
        image = Image.open(self.images_folder + str(image_index)).convert("RGB")
        pixel_values = self.feature_extractor(images=image, return_tensors="pt")['pixel_values']

        # T5 Input
        inputs = self.tokenizer(input, max_length=391, padding="max_length", truncation=True, return_tensors='pt')
        batch_size, sq_length = inputs['attention_mask'].size()
        inputs['attention_mask'] = torch.cat((torch.ones((batch_size, 121)), inputs['attention_mask']), dim=1)
        inputs['pixel_values'] = pixel_values

        # T5 outputs
        outputs = self.tokenizer(str(output), padding="max_length", return_tensors='pt')
        inputs['labels'] = outputs.input_ids
        #inputs['decoder_attention_mask'] = outputs['attention_mask']
        #inputs['decoder_inputs_embeds '] = self.t5_embeddings(outputs['input_ids'].to('cuda')).to('cpu')

        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)

        return inputs

    def __len__(self):
        return len(self.inputs)
