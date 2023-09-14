import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding

import argparse
import random
import os
import json

import numpy as np

import evaluate

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    set_seed(args.seed)

    model_id = "google/flan-t5-small"

    max_input_length = 1024
    max_target_length = 128

    phish_input = "Generate a phishing email"
    non_phish_input = "Generate a non-phishing email"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def preprocess_function(examples):
        labels = examples['Label']

        #use the labels as the input (as if asked to generate a phishing email)
        model_inputs = tokenizer([phish_input if label == '1' else non_phish_input for label in labels], max_length=max_input_length, truncation=True)

        #use the body as the label (the type of email we're trying to produce)
        labels = tokenizer(examples['Body'], max_length=max_input_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    if args.load_pt_from_file is not None:
        tokenized_datasets = torch.load(args.load_pt_from_file)
    else:
        data = torch.load(args.data_path)
        tokenized_datasets = data.map(preprocess_function, batched=True)

        try:
            torch.save(tokenized_datasets, f"{args.save_path}gen_processed_dataset_dict_qlora.pt")
        except Exception as e:
            print(f"Failed to save:\n{e}\n")

    train = tokenized_datasets['train']
    eval = tokenized_datasets['validation']
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map={"":0})

    # Define LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    from transformers import DataCollatorForSeq2Seq

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )
    
    #Accuracy Metrics
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{args.save_path}training_checkpoints_t5_qlora_new",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=1e-3, # higher learning rate
        num_train_epochs=5,
        logging_dir=f"{args.save_path}training_checkpoints_t5_qlora_logs",
        report_to="wandb",
        evaluation_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=200,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train,
        eval_dataset=eval,
        compute_metrics=compute_metrics,
    )

    trainer.train()#resume_from_checkpoint=True)

    trainer.save_model(f"{args.save_path}t5_qlora.model")

    results = trainer.evaluate()

    with open(f"{args.results_path}/t5_qlora.results", "w") as f:
        f.write(json.dumps(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training script for baseline model.")
    parser.add_argument("--model_name_or_path", type=str, default='bert-base-cased')
    parser.add_argument("--data_path", type=str, default='/mnt/parscratch/users/username/')
    parser.add_argument("--results_path", type=str, default='.')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_path", type=str, default='/mnt/parscratch/users/username/')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_pt_from_file", type=str, help="Skip preprocessing and load dataset from file")
    args = parser.parse_args()
    main(args)