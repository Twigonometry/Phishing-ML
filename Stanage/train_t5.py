import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

import argparse
import numpy as np
import evaluate
import json

def main(args):
    data = torch.load(args.data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    max_input_length = 1024
    max_target_length = 128

    phish_input = "Generate a phishing email"
    non_phish_input = "Generate a non-phishing email"

    def preprocess_function(examples):
        labels = examples['Label']

        #use the labels as the input (as if asked to generate a phishing email)
        model_inputs = tokenizer([phish_input if label == '1' else non_phish_input for label in labels], max_length=max_input_length, truncation=True)

        #use the body as the label (the type of email we're trying to produce)
        labels = tokenizer(examples['Body'], max_length=max_input_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = data.map(preprocess_function, batched=True)

    train = tokenized_datasets['train']
    test = tokenized_datasets['validation']

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    args = Seq2SeqTrainingArguments(
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True,
        save_steps=70650,
        logging_steps=500,
        output_dir=f"{args.save_path}training_checkpoints_train_t5",
        report_to='wandb'
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train,
        eval_dataset=test,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model(f"{args.save_path}train_t5.out")

    results = trainer.evaluate()

    with open(f"{args.results_path}/train_t5.results", "w") as f:
        f.write(json.dumps(results))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training script for T5.")
    parser.add_argument("--model_name_or_path", type=str, default='t5-small')
    parser.add_argument("--data_path", type=str, default='/mnt/parscratch/users/username/')
    parser.add_argument("--results_path", type=str, default='.')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_path", type=str, default='/mnt/parscratch/users/username/')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)