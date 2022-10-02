

This repo is adapted from [VL-T5](https://github.com/j-min/VL-T5)
# Training the VL-T5 Model

In order to train the VL-T5 model on any ChartQA dataset, you need to: 
* Prepare the training, validation, and test csv files (e.g., [example-csv-file](https://github.com/vis-nlp/ChartQA/blob/main/Figures%20and%20Examples/T5%20and%20VL-T5%20Input%20File%20Examples.csv)). The Input Column should contain the question and flatenned data table. The Output column should contain the final answer. 
* Organize your data folder in the following structure:

```
├── data                   
│   ├── train   
│   │   ├── data.csv
│   │   ├── features
│   │   │   ├── chart1_name.json
│   │   │   ├── chart2_name.json
│   │   │   ├── ...
│   └── validation  
│   │   ├── data.csv
│   │   ├── features
│   │   │   ├── chart1_name.json
│   │   │   ├── chart2_name.json
│   │   │   ├── ...
```
 <strong>Note:</strong> The features json files names should match the "Image Index" column values in the data.csv file. 

* Download the last pretrained checkpoint from the original VL-T5 repo ([VL-T5 Checkpoint](https://drive.google.com/drive/folders/12Acv2YLQSxgrx_-4mahUvqNikcz7XfPi)).

* Update the following hyperparameters in the ([VQA_VLT5.sh](https://github.com/vis-nlp/ChartQA/blob/main/Models/VL-T5/scripts/VQA_VLT5.sh))
  * src_folder: the path to your data folder
  * load: the path to your VL-T5 model. 
  * output: your desired output directory path
  * any other hyperparameters you want. 
  
* Run the following command with your prefered hyperparameters.
```
bash scripts/VQA_VLT5.sh gpus_number
```

# Prediction
You need to first prepare your data directory as mentioned above and update the hyperparameters in the inference script ([VQA_VLT5_inference.sh](https://github.com/vis-nlp/ChartQA/blob/main/Models/VL-T5/scripts/VQA_VLT5_inference.sh)) as described above, especially "load" which is the path to your trained model. You can then run the following command

```
bash scripts/VQA_VLT5_inference.sh gpus_number
```

 <strong>Note:</strong> The metric in this code is the exact accuracy which is different from the relaxed accuracy measure described in the paper. Hence, you will still need to evaluate the generated predictions using the relaxed accuracy. 
 
 
# Generating the Charts Visual Features Json Files
The code is adapted from [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html)
## Training Mask RCNN Model
To train the Mask RCNN model, you can run the [train.py](https://github.com/vis-nlp/ChartQA/blob/main/Models/VL-T5/Mask-RCNN/train.py) script as follows:

```
python train.py --train-data-dict data/train_dict.pickle --valid-data-dict data/val_dict.pickle --train-images data/images_train/ --valid-images data/images_validation/ --output-dir /content/output/ --ITERS 16000 --batch-size 2
```

You can also train the model in Colab by running the Training Section in the [Mask RCNN - ChartQA Colab Notebook](https://github.com/vis-nlp/ChartQA/blob/main/Models/VL-T5/Mask-RCNN/Mask_RCNN_ChartQA.ipynb).  

Your data directory should have the following structure:
```
├── data                 
│   ├── train_dict.pickle   
│   ├── val_dict.pickle   
│   ├── images_train           
│   │   ├── chart1_name.png
│   │   ├── chart2_name.png
│   │   ├── ...
│   ├── images_validation           
│   │   ├── chart1_name.png
│   │   ├── chart2_name.png
│   │   ├── ...

```
Moreover, the train_dict.pickle and val_dict.pickle files have the following structure:
```
a dictionary of img_names and img_data
 img_name: name of the image
 img_data: a dictionary containing the bounding boxes and the masks.
   bboxes: a list of tuples (class_name [x0, y0, x1, y1])
   masks: a list of lists (e.g., for a rectangle [x0, y0, x1, y0, x1, y1, x0, y1, x0, y0]) # for other shapes like pie, please refer to the Appendix of the paper. 
```
## Inference (generating the visual features)
In order to generate the visual features using your trained Mask RCNN model, you can also use the [Mask RCNN - ChartQA Colab Notebook](https://github.com/vis-nlp/ChartQA/blob/main/Models/VL-T5/Mask-RCNN/Mask_RCNN_ChartQA.ipynb). You first need to update the following parameters with your desired values:
```
trained_model_path = "model_final_all.pth"
images_path = "/content/data/images_validation/"
save_path = "/content/output_feats/"
CLASSES_NUM = 15
```
After that, you can run all the cells under the Inference Section in the colab notebook. 
** If you don't want to waste time training a new model, you can generate the visual features of the chart images using [our trained Mask RCNN model](https://drive.google.com/file/d/1UgXmggU6P-d9GE6tdZrZsfc61ZuGcvSA/view?usp=sharing) which we finetuned on all the datasets used in our paper (FigureQA, PlotQA, DVQA, ChartQA). **
