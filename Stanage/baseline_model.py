# Imports
import pandas as pd
import numpy as np
import json

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from datasets import DatasetDict

import argparse

import evaluate

import torch
import random
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    # == Utility Functions ==
            
    set_seed(args.seed)

    # == Load the data ==
    
    data = torch.load(args.data_path)

    #instantiate pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    #apply tokenizer
    data = data.map(
            lambda x: {'Label': [int(x_i) for x_i in x['Label']], **tokenizer(x['Body'], truncation=True)},
            batched=True,
            num_proc=os.cpu_count() // 2,
        ).remove_columns('Body').rename_column('Label', 'labels')

    print("mapped")

    train = data['train']
    test = data['validation']

    #Define Model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=2)

    collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                    padding=True,
                                    pad_to_multiple_of=8,                   # Better utilisation of A100 tensor cores
                                    return_tensors="pt")

    #Hyperparameters
    training_args = TrainingArguments(output_dir=f"{args.save_path}training_checkpoints_baseline_model",
                                    evaluation_strategy="epoch",
                                    report_to='wandb',
                                    save_steps=70650,
                                    logging_steps=500,
                                    per_device_train_batch_size=args.batch_size,)

    #Accuracy Metrics
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    #Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        compute_metrics=compute_metrics,
        data_collator=collator,
    )

    print("training")

    trainer.train()

    trainer.save_model(f"{args.save_path}baseline_model.out")

    results = trainer.evaluate()

    with open(f"{args.results_path}/baseline_model.results", "w") as f:
        f.write(json.dumps(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training script for baseline model.")
    parser.add_argument("--model_name_or_path", type=str, default='bert-base-cased')
    parser.add_argument("--data_path", type=str, default='/mnt/parscratch/users/username/')
    parser.add_argument("--results_path", type=str, default='.')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_path", type=str, default='/mnt/parscratch/users/username/')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)