# Imports
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

import evaluate

# == Utility Functions ==

def load_processed_enron():
    dfEnronFull = pd.read_csv('./Processed-Datasets/Enron-Bodies/emails.csv', usecols=['message'])
    dfEnronFull["id"] = dfEnronFull.index + 1
    dfEnronFull["Label"] = 0

    return dfEnronFull

# == Load the data ==

spam_col_names = ['id', 'Body', 'Label']
dfSA = pd.read_csv('./kaggle-datasets/Email-Spam-Dataset/completeSpamAssassin.csv', names=spam_col_names)
dfEnron = pd.read_csv('./kaggle-datasets/Email-Spam-Dataset/enronSpamSubset.csv', names=spam_col_names).tail(-1)
dfLing = pd.read_csv('./kaggle-datasets/Email-Spam-Dataset/lingSpam.csv', names=spam_col_names)

# Load and rename Enron
dfEnronFull = load_processed_enron()
dfEnronFull = dfEnronFull.rename(columns={"message":"Body"})

# Combine Datasets

dfAll = pd.concat([dfSA, dfEnron, dfLing, dfEnronFull]).tail(-1).drop(columns=['message']).dropna()

# == Define Regexes ==

url_pat = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'

email_pat = '^[\w\-\.]+@([\w-]+\.)+[\w-]{2,4}$'

# Replace URLs

dfAll = dfAll.replace(to_replace=url_pat, value='<URL>', regex=True)

# Replace Emails

dfAll = dfAll.replace(to_replace=email_pat, value='<EMAIL>', regex=True)

print(dfAll)
# print(type(dfAll['Body']))

# == Scikit Tokenizer

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(list(dfAll['Body'].values))

#convert from sparse matrix
X = pd.DataFrame.sparse.from_spmatrix(X)

print("Transformed")

y = dfAll.iloc[:, -1]

# == Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Split")

train = pd.concat([X_train, y_train])
test = pd.concat([X_test, y_test])

small_eval = test.shuffle(seed=42).select(range(1000))

# Train BERT Classifier

print("Training")

#Define Model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

#Hyperparameters
training_args = TrainingArguments(output_dir="train_checkpoints", evaluation_strategy="epoch")

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
    eval_dataset=small_eval,
    compute_metrics=compute_metrics,
)

trainer.train()