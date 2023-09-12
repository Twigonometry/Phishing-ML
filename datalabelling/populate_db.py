# Imports
import pandas as pd
import pymongo
from tqdm import tqdm

myclient = pymongo.MongoClient("mongodb://127.0.0.1:27017/my_database")
mydb = myclient["mydatabase"]
mycol = mydb["emails"]

# == Utility Functions ==

def load_processed_enron():
    dfEnronFull = pd.read_csv('../Processed-Datasets/Enron-Bodies/emails.csv', usecols=['message'])
    dfEnronFull["id"] = dfEnronFull.index + 1
    dfEnronFull["Label"] = 0

    return dfEnronFull

# == Load the data ==

spam_col_names = ['id', 'Body', 'Label']
dfSA = pd.read_csv('../kaggle-datasets/Email-Spam-Dataset/completeSpamAssassin.csv', names=spam_col_names).sample(200)
dfEnron = pd.read_csv('../kaggle-datasets/Email-Spam-Dataset/enronSpamSubset.csv', names=spam_col_names).tail(-1).sample(100)
dfLing = pd.read_csv('../kaggle-datasets/Email-Spam-Dataset/lingSpam.csv', names=spam_col_names).sample(100)

# Load and rename Enron
dfEnronFull = load_processed_enron()
dfEnronFull = dfEnronFull.rename(columns={"message":"Body"}).sample(400)

# print(dfEnronFull.head())

# Combine Datasets

dfAll = pd.concat([dfSA, dfEnron, dfLing, dfEnronFull])
print(dfAll.head())
dfAll.dropna()

# Load all items into DB

for message in tqdm(dfAll['Body']):
    mycol.insert_one({"body": message})