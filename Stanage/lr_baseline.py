# Imports
import pandas as pd
import numpy as np
import json
import gensim
import gensim.downloader as gensim_api
from gensim.models.word2vec import Word2Vec as w2v
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import torch
import argparse
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    # == Utility Functions ==
            
    set_seed(42)

    data = torch.load(args.data_path)

    if args.from_dataframe:
        train_data, test_data = data.train_test_split(test_size=0.2).values()
    else:
        train_data = data['train']
        test_data = data['test']

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    iter=5
    min_count=1
    window=8

    representation = w2v(train_df['Body'], vector_size=300, window=window, min_count=min_count, sg=1, epochs=iter, callbacks=[])

    train(train_df, test_df, representation, args)

def get_average_doc_embedding(representation, document):
    embeddings = []
    for word in document.split(" "):
        if word in representation.wv:
            embeddings.append(representation.wv[word])
    
    if len(embeddings) > 0:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(representation.vector_size)
    
def get_numerical_rep(data, representation):
    rep = []

    for doc in data['Body']:
        rep.append(get_average_doc_embedding(representation, doc))

    return rep
    
def train(train_df, test_df, representation, args):
    train_embeddings = get_numerical_rep(train_df, representation)
    test_embeddings = get_numerical_rep(test_df, representation)

    model = LogisticRegression(random_state=0).fit(train_embeddings, train_df['Label'])

    torch.save(model, f"{args.data_path}lr_baseline_model_checkpoint")

    predicted = model.predict(test_embeddings)

    accuracy = metrics.accuracy_score(test_df['Label'], predicted)
    matrix = metrics.confusion_matrix(test_df['Label'], predicted)
    tn, fp, fn, tp = matrix.ravel()

    print(accuracy)
    print(f"{tn} {fp} {fn} {tp}")

    with open(args.save_path, 'w') as f:
        f.write(accuracy)
        f.write(f"{tn} {fp} {fn} {tp}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training script for log reg.")
    parser.add_argument("--data_path", type=str, default='/mnt/parscratch/users/username/')
    parser.add_argument("--save_path", type=str, default='/mnt/parscratch/users/username/')
    parser.add_argument("--from_dataframe", action="store_true")
    args = parser.parse_args()
    main(args)