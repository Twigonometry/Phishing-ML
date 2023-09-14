import argparse
import torch
from transformers import(
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    TextClassificationPipeline,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import pandas as pd
from datasets import Dataset, DatasetDict
import os
import evaluate
import numpy as np
import json

metric = evaluate.load("accuracy")

phish_input = "Generate a phishing email"
non_phish_input = "Generate a non-phishing email"

def preprocess_t5(examples, tokenizer, max_input_length=1024):
        labels = examples['Label']

        #use the labels as the input (as if asked to generate a phishing email)
        #labels have already been replaced with phish_input/non_phish_input
        model_inputs = tokenizer(labels, max_length=max_input_length, truncation=True)

        #use the body as the label (the type of email we're trying to produce)
        labels = tokenizer(examples['Body'], max_length=max_input_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def dis_fine_tune(emails, labels, model, tokenizer, iter, args, max_iter):
    print(f"Fine tune dis iter {iter}")
    
    #combine the emails and labels into a dataset
    df = pd.DataFrame(
    {'Body': emails,
     'Label': labels,
    })

    #load and process data

    df.reset_index(drop=True, inplace=True)
    df = df.drop_na()

    ds = Dataset.from_pandas(df)

    train, test = ds.train_test_split(test_size=0.2).values()

    ds_dict = DatasetDict()
    ds_dict['train'] = train
    ds_dict['test'] = test

    data = ds_dict.map(
            lambda x: {'Label': [int(x_i) for x_i in x['Label']], **tokenizer(x['Body'], truncation=True)},
            batched=True,
            num_proc=os.cpu_count() // 2,
        ).remove_columns('Body').rename_column('Label', 'labels')
    
    train = data['train']
    test = data['test']

    collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                    padding=True,
                                    pad_to_multiple_of=8,                   # Better utilisation of A100 tensor cores
                                    return_tensors="pt")
    
    training_args = TrainingArguments(output_dir=f"{args.save_path}adversarial_dis_{iter}",
                                    evaluation_strategy="epoch",
                                    save_steps=500,
                                    logging_steps=500,
                                    per_device_train_batch_size=args.batch_size,)
    
    #define and run trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        compute_metrics=compute_metrics,
        data_collator=collator,
    )

    trainer.train()

    results = trainer.evaluate()

    with open(f"{args.results_path}/adversarial_dis_{iter}.results", "w") as f:
        f.write(json.dumps(results))

    #output if done
    if iter == max_iter:
        trainer.save_model(f"{args.save_path}dis_fine_tuned.out")

    return trainer.model

def gen_fine_tune(emails, labels, model, tokenizer, iter, args, max_iter):
    """fine tunes the generator"""
    print(f"Fine tune gen iter {iter}")

    print(f"Example email before fine tuning: {emails[0]}")

    #load the data

    df = pd.DataFrame(
    {'Body': emails,
     'Label': labels,
    })

    df.reset_index(drop=True, inplace=True)
    df = df.drop_na()

    ds = Dataset.from_pandas(df)

    #split the data

    train, test = ds.train_test_split(test_size=0.2).values()

    ds_dict = DatasetDict()
    ds_dict['train'] = train
    ds_dict['test'] = test

    #preprocess the data

    data = ds_dict.map(preprocess_t5, batched=True)
    
    train = data['train']
    test = data['test']

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    #training arguments
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{args.save_path}training_checkpoints_t5_qlora_new",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=1e-3, # higher learning rate
        num_train_epochs=5,
        logging_dir=f"{args.save_path}training_checkpoints_t5_qlora_logs",
        evaluation_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=200,
    )

    #define and run trainer

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train,
        eval_dataset=test,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = trainer.evaluate()

    with open(f"{args.results_path}/adversarial_dis_{iter}.results", "w") as f:
        f.write(json.dumps(results))

    #output if done

    if iter == max_iter:
        trainer.save_model(f"{args.save_path}gen_fine_tuned.out")

    return trainer.model

def main(args):
    #load in pretrained models from file/huggingface hub
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(args.gen_model_path)
    dis_model = AutoModelForSequenceClassification.from_pretrained(
        args.dis_model_path, num_labels=2
    )

    gen_tokenizer = AutoTokenizer.from_pretrained(args.gen_tokenizer_name)
    dis_tokenizer = AutoTokenizer.from_pretrained(args.dis_tokenizer_name)

    learning_rate = 1e-3

    #TODO: manual wandb?

    max_iter = args.epochs + 1

    #for each iteration, generate data and train
    for i in range(1, max_iter):
        print(f"Epoch {i}")

        #generate emails_per_epoch fresh emails
        inputs = gen_tokenizer(phish_input, return_tensors="pt")
        train_outputs = []
        test_outputs = []
        for i in range(0, 2 * args.emails_per_epoch):
            #generate using gen_model
            gen_outputs = gen_model.generate(**inputs)

            #add half the data to the list of emails for training, half to that used for fine tuning gen
            #batch_decode turns it back into a string
            if i % 2 == 0:
                train_outputs.append(gen_tokenizer.batch_decode(gen_outputs, skip_special_tokens=True))
            else:
                test_outputs.append(gen_tokenizer.batch_decode(gen_outputs, skip_special_tokens=True))

        #fine tune discriminator using the generated phishing emails
        train_labels = ["1" for email in train_outputs]
        dis_model = dis_fine_tune(train_outputs, train_labels, dis_model, dis_tokenizer, i, args, max_iter)

        #classify using gen_model, return the labels
        #can maybe do in one step if can access the labels assigned during training
        pipe = TextClassificationPipeline(model=dis_model, tokenizer=dis_tokenizer)

        labels = []
        for output in test_outputs:
            prediction = pipe(output)
            label = prediction[0]['label']
            labels.append(label)

        #relabel correctly classified ones as non-phishing and misclassified ones as phishing
        new_labels = [non_phish_input if label == 'LABEL_1' else phish_input for label in labels]

        #and fine tune generator
        gen_model = gen_fine_tune(test_outputs, new_labels, gen_model, gen_tokenizer, i, args, max_iter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training script for Adversarial Learning.")
    parser.add_argument("--gen_model_path", type=str, default="google/flan-t5-small", help="Path to pretrained generative model")
    parser.add_argument("--dis_model_path", type=str, default="/mnt/parscratch/users/harcoded/baseline_model.out", help="Path to pretrained discriminator model")
    parser.add_argument("--gen_tokenizer_name", type=str, default="google/flan-t5-small", help="Name of AutoTokenizer to use for generative model")
    parser.add_argument("--dis_tokenizer_name", type=str, default="bert-base-cased", help="Name of AutoTokenizer to use for discriminator model")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--emails_per_epoch", type=int, help="How many emails to generate per epoch")
    parser.add_argument("--max_length", type=int, default=250, help="Max length of generated emails")
    parser.add_argument("--save_path", type=str, default="/mnt/parscratch/users/hardcoded/", help="Where to save models")
    args = parser.parse_args()

    main(args)