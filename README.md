# ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning

* Authors: Ahmed Masry, Do Long, Jia Qing Tan, Shafiq Joty, Enamul Hoque
* Paper Link: [ChartQA](https://aclanthology.org/2022.findings-acl.177/)
* **[NEW]** If you are looking for powerful Chart Models, checkout out our following models:
    * [UniChart](https://github.com/vis-nlp/UniChart)
        * This is a very lightweight model (140M parameters only) that works really well for ChartQA, Chart to Table, Chart Summarization, and Open-ended QA.
    * [ChartInstruct](https://github.com/vis-nlp/ChartInstruct)
        * This is our Chart Large Language Model that builds upon LLaVA and LLama2 (7B parameters)! It's a general purpose chart understanding model that works for a variety of tasks!
    * **Both Models are very easy to run with a few lines of code and they also have public web demoes! Check their github repos for details**

## Updates
* Added VisionTaPas Model
* Added the Mask-RCNN training and inference codes to generate the visual features for VL-T5
* Added the full ChartQA dataset (including the bounding boxes annotations)
* Added T5 and VL-T5 models codes along with the instructions. 
* Added the first version of the ChartQA dataset (does not have the annotations folder)

## ChartQA Dataset
### First Version (does not have the annotations folder)
The ChartQA dataset is available in the ChartQA Dataset folder in this repository. 

### Full Version (with the annotations folder)
The full ChartQA dataset (including the annotations) can be downloaded from the following huggingface dataset: [Full ChartQA Dataset](https://huggingface.co/datasets/ahmed-masry/ChartQA). The dataset has the following structure:

```
├── ChartQA Dataset                   
│   ├── train   
│   │   ├── train_augmented.json # ChartQA-M (machine-generated) questions/answers. 
│   │   ├── train_human.json     # ChartQA-H (human-authored) questions/answers. 
│   │   ├── annotations           # Chart Images Annotations Folder
│   │   │   ├── chart1_name.json
│   │   │   ├── chart2_name.json
│   │   │   ├── ...
│   │   ├── png                   # Chart Images Folder
│   │   │   ├── chart1_name.png
│   │   │   ├── chart2_name.png
│   │   │   ├── ...
│   │   ├── tables                # Underlying Data Tables Folder
│   │   │   ├── chart1_name.csv
│   │   │   ├── chart2_name.csv
│   │   │   ├── ...
│   └── val  
│   │   │   ...
│   │   │   ...
│   └── test  
│   │   │   ...
│   │   │   ...
│   │   |   ...
```

 <strong>Note:</strong> In order to produce the annotations (e.g., bounding boxes) for the charts, we processed the SVG files of these charts automatically. However, some of the SVG files were corrupt/noisy/missing, so the provided annotations in this dataset are a bit noisy. Moreover, the Pew Research Centre chart images didn't have any SVG files when we crawled them. That's why we had to manually annotate them and use some heuristics to accelerate the annotation process. 
 
Each annotation json file has the following format (similar to [PlotQA](https://github.com/NiteshMethani/PlotQA/blob/master/PlotQA_Dataset.md) and [FigureQA](https://www.microsoft.com/en-us/research/project/figureqa-dataset/) datasets):
```
models: a list of dictionaries where each dictionary contains the following keys:
    **For bar and line charts**
      name: The Legend Label of the data points (bars, line).
      color: Color of the data points (bars, line). 
      bboxes: Bounding boxes of the data points (bars, line segments)
      x: x-value of the datapoints.
      y: y-value of the datapoints.
     ** Pie Charts **
      name: The label of the pie slice
			color: Color of the pie slice.
			bbox: Bounding box of the pie slice
			value: Value of the pie slice
			text_label: Text label of the pie slice
			text_bbox: Bounding box of the text label
      points: Coordinates of the start/end/center points of the pie slice. 

type: Chart Type (v_bar, h_bar, line, pie).

general_figure_info: It is a dictionary containng the following keys-
		title: Bounding box and the text corresponding to the title of the plot.
		x_axis: Bounding boxes, axis labels corresponding to the x-axis of the chart image.
		y_axis: Bounding boxes, axis labels corresponding to the y-axis of the chart image.
		legend: Bounding boxes, axis labels corresponding to the legend of the chart image.
		figure_info: Bounding box corresponding to the plot area of the chart image.
```
## Models

### VL-T5
Please refer to [VL-T5](https://github.com/vis-nlp/ChartQA/tree/main/Models/VL-T5)

### T5 
Please refer to [T5](https://github.com/vis-nlp/ChartQA/tree/main/Models/T5)

### VisionTapas
Please refer to [VisionTapas](https://github.com/vis-nlp/ChartQA/tree/main/Models/VisionTapas)

# Contact
If you have any questions about this work, please contact **Ahmed Masry** using the following email address: **amasry17@ku.edu.tr**.
Please note that my school email which was mentioned in the paper (**masry20@yorku.ca**) has been deactivated since I have already graduated. 

# Reference
Please cite our paper if you use our models or dataset in your research. 

```
@inproceedings{masry-etal-2022-chartqa,
    title = "{C}hart{QA}: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning",
    author = "Masry, Ahmed  and
      Long, Do  and
      Tan, Jia Qing  and
      Joty, Shafiq  and
      Hoque, Enamul",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.177",
    doi = "10.18653/v1/2022.findings-acl.177",
    pages = "2263--2279",
}
```
