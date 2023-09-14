import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

username = 'hardcoded'

data_path = f'/mnt/parscratch/users/{username}/Data/'

def load_dfs():
    spam_col_names = ['Body', 'Label']
    dfSA = pd.read_csv(f'{data_path}completeSpamAssassin.csv', names=spam_col_names)
    dfEnron = pd.read_csv(f'{data_path}enronSpamSubset.csv', names=spam_col_names).tail(-1)
    dfLing = pd.read_csv(f'{data_path}lingSpam.csv', names=spam_col_names)

    dfmbox = pd.read_csv(f'{data_path}phishing_mbox.csv', usecols=['Body'])
    dfmbox['Label'] = 1

    dfEnronFull = pd.read_csv(f'{data_path}emails.csv', usecols=['message'])

    #extract text from enron and adjust columns
    dfEnronFull['message'] = dfEnronFull['message'].str.split("\n\n", n=1).str[1]
    # dfEnronFull["id"] = dfEnronFull.index + 1
    dfEnronFull["Label"] = 0
    dfEnronFull = dfEnronFull.rename(columns={"message":"Body"})

    dfAll = pd.concat([dfSA, dfEnron, dfLing, dfEnronFull, dfmbox]).tail(-1).dropna()

    dfAll['Body'] = dfAll['Body'].astype(str)
    dfAll['Label'] = dfAll['Label'].astype(str)

    dfAll.reset_index(drop=True, inplace=True)

    # dfAll.to_csv(f'{data_path}all_data.csv', columns=['Body', 'Label'])

    return dfAll

def split_df(df, percent):
    train_size = int(percent * len(df))
    test_size = len(df) - train_size

    train, test = torch.utils.data.random_split(df, [train_size, test_size])

    return train, test

#combine all datasets
dfAll = load_dfs()

#cast to dataset object
ds = Dataset.from_pandas(dfAll)#, preserve_index=False)

#split into train, test, val

train, not_test = split_df(ds, 0.8)

val, test = split_df(not_test, 0.4)

#create DatasetDict
ds = DatasetDict()

ds['train'] = train
ds['validation'] = val
ds['test'] = test

# Train Tokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(train['Body'])

def preprocess(data):
    # == Define Regexes ==
    url_pat = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
    email_pat = '^[\w\-\.]+@([\w-]+\.)+[\w-]{2,4}$'

    # Replace URLs
    data['Body'] = data['Body'].replace(to_replace=url_pat, value='<URL>', regex=True)

    # Replace Emails
    data['Body'] = data['Body'].replace(to_replace=email_pat, value='<EMAIL>', regex=True)

    return data

ds.map(preprocess)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

ds = ds.map(
        lambda x: tokenizer(x['Body'], truncation=True),
        batched=True
    ).remove_columns('Body').rename_column('Label', 'labels')

torch.save(ds, f'{data_path}dataset_dict')