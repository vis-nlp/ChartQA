

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
Coming Soon
