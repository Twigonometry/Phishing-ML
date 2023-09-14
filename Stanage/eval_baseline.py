from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

import argparse
import os
import torch
import numpy as np

parser = argparse.ArgumentParser("Evaluate baseline")
parser.add_argument("--model_name_or_path", type=str, default='bert-base-cased')
parser.add_argument("--data_path", type=str, default='/mnt/parscratch/users/username/')
args = parser.parse_args()

data = torch.load(args.data_path)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    #apply tokenizer
data = data.map(
        lambda x: {'Label': [int(x_i) for x_i in x['Label']], **tokenizer(x['Body'], truncation=True)},
        batched=True,
        num_proc=os.cpu_count() // 2,
    ).remove_columns(['Body', '__index_level_0__']).rename_column('Label', 'labels')

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

test_data = data['test']

try:
    torch.save(test_data, f"{args.data_path}eval_baseline_test_data_checkpoint.out")
except Exception as e:
    print(e)

model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)

from transformers import pipeline
from datasets import load_dataset
from evaluate import evaluator
import evaluate

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)

for data in test_data:
    prediction = pipe(prediction = pipe)

# Get predictions
y_pred = model.predict([input_ids_test, attention_mask_test, token_type_ids_test])
y_pred_proba = [float(x[1]) for x in tf.nn.softmax(y_pred.logits)]
y_pred_label = [0 if x[0] > x[1] else 1 for x in tf.nn.softmax(y_pred.logits)]


# Evaluate the model
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

print("Confusion Matrix : ")
print(confusion_matrix(labels_test, y_pred_label))

print("ROC AUC score : ", round(roc_auc_score(labels_test, y_pred_proba), 3))

print("Average Precision score : ", round(average_precision_score(labels_test, y_pred_proba), 3))

# return accuracy.compute(predictions=predictions, references=labels)

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

pipe = pipeline("text-classification", model=model)
task_evaluator = evaluator("text-classification")
results = task_evaluator.compute(
    model_or_pipeline=pipe,
    data=test_data,
    tokenizer=tokenizer,
    label_mapping={"NEGATIVE": 0, "POSITIVE": 1}
)

print(results)