import pandas as pd
import torch
from datasets import Dataset, DatasetDict
import re
import argparse

def split_dataset(data, train_size=0.8, test_size=0.12):
    """split data into train, test, validation"""

    train, not_train = data.train_test_split(test_size=(1-train_size)).values()

    val, test = not_train.train_test_split(test_size=test_size).values()

    return train, val, test

def preprocess(data):
    # == Define Regexes ==
    url_pat = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
    email_pat = r'^[\w\-\.]+@([\w-]+\.)+[\w-]{2,4}$'

    input = data['Body']

    no_url = re.sub(url_pat, '<URL>', input)
    no_email = re.sub(email_pat, '<EMAIL>', no_url)

    data['Body'] = no_email
    
    return data

def main(args):
    data_path = args.data_path

    #read the data and reset indices
    df = pd.read_pickle(data_path)
    df.reset_index(drop=True, inplace=True)

    #remove data with the column name as the row
    df = df.drop(df[df['Label'] == 'Label'].index)

    #cast to dataset object
    ds = Dataset.from_pandas(df)

    train, val, test = split_dataset(ds)

    #create DatasetDict for generator
    ds_dict = DatasetDict()

    ds_dict['train'] = train
    ds_dict['validation'] = val
    ds_dict['test'] = test

    ds_dict.map(preprocess)

    torch.save(ds_dict, args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Save dataset without splitting into gen/dis.")
    parser.add_argument("--data_path", type=str, default='/mnt/parscratch/users/username/')
    parser.add_argument("--save_path", type=str, default="./full_dataset", help="Where to save data")
    parser.add_argument("--train_size", type=int, default=0.8, help="Percentage size of train dataset")
    parser.add_argument("--test_size", type=int, default=0.12, help="Percentage size of test dataset")
    args = parser.parse_args()

    main(args)