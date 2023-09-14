import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import re

username = 'hardcoded'

data_path = f'/mnt/parscratch/users/{username}/'

df = pd.read_pickle(f'{data_path}dfAll_for_liam/data.pkl')
df.reset_index(drop=True, inplace=True)
df = df.drop(df[df['Label'] == 'Label'].index)

#cast to dataset object
ds = Dataset.from_pandas(df)

#split into data for generator and discriminator

gen, dis = ds.train_test_split(test_size=0.5).values()

#split into train, test, val
def split_dataset(data):
    train, not_train = data.train_test_split(test_size=0.2).values()

    val, test = not_train.train_test_split(test_size=0.6).values()

    return train, val, test

gen_train, gen_val, gen_test = split_dataset(gen)
dis_train, dis_val, dis_test = split_dataset(dis)

#create DatasetDict for generator
ds_gen = DatasetDict()

ds_gen['train'] = gen_train
ds_gen['validation'] = gen_val
ds_gen['test'] = gen_test

#create DatasetDict for discriminator
ds_dis = DatasetDict()

ds_dis['train'] = dis_train
ds_dis['validation'] = dis_val
ds_dis['test'] = dis_test

def preprocess(data):
    # == Define Regexes ==
    url_pat = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
    email_pat = r'^[\w\-\.]+@([\w-]+\.)+[\w-]{2,4}$'

    input = data['Body']

    no_url = re.sub(url_pat, '<URL>', input)
    no_email = re.sub(email_pat, '<EMAIL>', no_url)

    data['Body'] = no_email
    
    return data

ds_dis.map(preprocess)
ds_gen.map(preprocess)

#save
torch.save(ds_gen, f'{data_path}gen_dataset_dict')
torch.save(ds_dis, f'{data_path}dis_dataset_dict')