import pandas as pd
import numpy as np
import torch
import argparse
import random
from gensim.models.word2vec import Word2Vec as w2v
import torch
import re

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

label_cols = ['rating.authoritative', 'rating.threatening',
       'rating.rewarding', 'rating.unnatural', 'rating.emotional',
       'rating.provoking', 'rating.timesensitive', 'rating.imperative']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_average_doc_embedding(representation, document):
    embeddings = []
    for word in document.split(" "):
        if word in representation.wv:
            embeddings.append(representation.wv[word])
    
    if len(embeddings) > 0:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(representation.vector_size)

def get_numerical_rep(data, representation, bodyCol='body'):
    rep = []

    for doc in data[bodyCol]:
        rep.append(get_average_doc_embedding(representation, doc))

    return rep

def train_word_rep(data):
    iter=5
    min_count=1
    window=8

    #instantiate word2vec
    representation = w2v(data['body'], vector_size=300, window=window, min_count=min_count, sg=1, epochs=iter, callbacks=[])

    return representation

def train_custom(data, representation, args):
    #get embeddings for body
    train_embeddings = get_numerical_rep(data, representation)

    #get the values of the labels we're trying to predict
    labels = data[label_cols].values

    #create classifier
    clf = MultiOutputClassifier(LogisticRegression()).fit(train_embeddings, labels)

    return clf

def label_custom(data, clf, representation, args):
    test_embeddings = get_numerical_rep(data, representation, bodyCol='Body')
    preds = clf.predict(test_embeddings)
    return preds

def label_vader(data):
    sid_obj = SentimentIntensityAnalyzer()

    scores = [(lambda x: [x['neg'], x['pos']])(sid_obj.polarity_scores(s)) for s in data['Body']]

    vader_df = pd.DataFrame(scores, columns=['neg', 'pos'])

    data['neg'] = vader_df['neg']
    data['pos'] = vader_df['pos']

def preprocess(row):
    # == Define Regexes ==
    url_pat = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
    email_pat = r'^[\w\-\.]+@([\w-]+\.)+[\w-]{2,4}$'

    no_url = re.sub(url_pat, '<URL>', row)
    no_email = re.sub(email_pat, '<EMAIL>', no_url)
    
    return no_email

def main(args):
    train_data = pd.read_csv(args.train_data_path)

    #'test' data is actually the full dataset - we want to add the labels to it
    #so that they can be passed into classifier
    if args.split_train_set:
        train_data, test_data = train_test_split(train_data, test_size=0.2)
    else:
        test_data = pd.read_pickle(args.test_data_path)
        test_data.reset_index(drop=True, inplace=True)
        test_data = test_data.drop(test_data[test_data['Label'] == 'Label'].index)

    set_seed(42)

    representation = train_word_rep(train_data)

    #train the custom dataset to predict labels
    clf = train_custom(train_data, representation, args)

    #label the rest of the data
    preds = label_custom(test_data, clf, representation, args)
    custom_df = pd.DataFrame({'authoritative': preds[:, 0],
                              'threatening': preds[:, 1],
                              'rewarding': preds[:, 2],
                              'unnatural': preds[:, 3],
                              'emotional': preds[:, 4],
                              'provoking': preds[:, 5],
                              'timesensitive': preds[:, 6],
                              'imperative': preds[:, 7]})
    
    test_data['authoritative'] = custom_df['authoritative']
    test_data['threatening'] = custom_df['threatening']
    test_data['rewarding'] = custom_df['rewarding']
    test_data['unnatural'] = custom_df['unnatural']
    test_data['emotional'] = custom_df['emotional']
    test_data['provoking'] = custom_df['provoking']
    test_data['timesensitive'] = custom_df['timesensitive']
    test_data['imperative'] = custom_df['imperative']

    if not args.no_vader:
        label_vader(test_data)

    if args.replace_tokens:
        test_data['Body'] = test_data['Body'].apply(preprocess, axis=1)

    #TODO: add NRC
    # too complicated to tokenize and look up - bottleneck
    #library does not work properly

    #save data to disk
    torch.save(test_data, args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Label data")
    parser.add_argument("--train_data_path")
    parser.add_argument("--test_data_path")
    parser.add_argument("--split_train_set", action='store_true', help="Use this in testing if want to split smaller train set rather than loading large test")
    parser.add_argument("--save_path")
    parser.add_argument("--replace_tokens", action='store_true')
    parser.add_argument("--no_vader", action='store_true')
    args = parser.parse_args()
    main(args)