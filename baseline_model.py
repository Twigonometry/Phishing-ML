# Imports
import pandas as pd
import re

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
dfEnronFull.rename(columns={"message":"Body"})

# Combine Datasets

dfAll = pd.concat([dfSA, dfEnron, dfLing, dfEnronFull])

# == Define Regexes ==

url_pat = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'

email_pat = '^[\w\-\.]+@([\w-]+\.)+[\w-]{2,4}$'

# Replace URLs

dfAll = dfAll.replace(to_replace=url_pat, value='<URL>', regex=True)

# Replace Emails

dfAll = dfAll.replace(to_replace=email_pat, value='<EMAIL>', regex=True)

# Print Head

print(dfAll.head())