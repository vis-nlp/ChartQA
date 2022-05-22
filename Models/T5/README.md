
# Training the T5 Model

In order to train the T5 model on any ChartQA dataset, you need to: 
* Prepare the training, validation, and test csv files (e.g., [example-csv-file](https://github.com/vis-nlp/ChartQA/blob/main/Figures%20and%20Examples/T5%20and%20VL-T5%20Input%20File%20Examples.csv)). The Input Column should contain the question and flatenned data table. The Output colum should contain the final answer. 
* Run the following command with your prefered hyperparameters.

```
python -m torch.distributed.run --nproc_per_node 1 run_T5.py \   
--model_name_or_path t5-base \   
--do_train \
--do_eval \
--do_predict \ 
--train_file /path-to-files/train.csv \
--validation_file /path-to-files/val.csv \
--test_file /path-to-files/test.csv \
--text_column Input \
--summary_column Output \
--source_prefix "" \
--output_dir /path-to-output/ \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=16 \
--predict_with_generate=True \
--learning_rate=0.0001 \
--num_beams=4 \
--num_train_epochs=30 \
--save_steps=2000 \
--eval_steps=2000 \
--evaluation_strategy steps \
--load_best_model \
--overwrite_output_dir \
--max_source_length=1024
```
