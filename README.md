# Phishing-ML
Experimental repository on phishing datasets for dissertation

# Installation

Requires python 3. Run:

```cmd
python -m pip install -r .\requirements.txt
```

or on unix

```bash
$ python3 -m pip install -r ./requirements.txt
```

You may first have to install pip on a fresh system:

```bash
$ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
$ python3 get-pip.py
```

If this is your first time using nltk, you may need to uncomment some of these statements and run the cell:

```python
#download wordnet for lemmatization
#uncomment appropriate line if you get error: "Resource wordnet not found.", "Resource punkt not found.", etc...

# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
```

these download the various resources used by NLTK in preprocessing.

The same goes for uncommenting this line

```python
# ! python -m spacy download en_core_web_sm
```

when setting up spacy.

If you receive this error when installing spacy packages:

```bash
$ python3 -m spacy download en_core_web_sm

...

TypeError: __init__() got an unexpected keyword argument 'no_args_is_help'
```

Run this:

```bash
$ pip3 install click --upgrade
```

You can now re-run the install command (either in command line or via notebook)

## Datasets

### Phishing Emails

The following three datasets are included within the repository:

Email-Spam-Dataset: https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset
- License: https://creativecommons.org/publicdomain/zero/1.0/

Phishing-Dataset-for-Machine-Learning: https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning
- License: https://creativecommons.org/licenses/by/4.0/

Web-page-Phishing-Detection-Dataset: https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset
- License: https://creativecommons.org/licenses/by/4.0/

### Enron

The full Enron dataset **must be manually downloaded** from the link below - extract emails.csv to the folder `kaggle-datasets/The-Enron-Email-Dataset`

The-Enron-Email-Dataset: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
- License: copyright original author

```bash
$ unzip archive.zip 
Archive:  archive.zip
  inflating: emails.csv
$ mkdir ~/Documents/Phishing-ML/kaggle-datasets/The-Enron-Email-Dataset
$ mv emails.csv ~/Documents/Phishing-ML/kaggle-datasets/The-Enron-Email-Dataset/emails.csv
```

#### Processing of Enron

Considerable effort went into combining the full Enron dataset with the elements from Email-Spam-Dataset that are labelled as spam. Unfortunately, due to the size of the dataset, this cannot be uploaded to GitHub. The method `mergeEnron()` can be called, after which the fully labelled Enron dataset can be loaded from the CSV.

This process is needed as all Enron emails need a label, but we cannot assume they are all non-spam. In fact, some have been labelled as spam; the goal is to consolidate the two datasets and transfer across the labels from Email-Spam-Dataset, but unfortunately the data formats are different so some processing had to be done to identify which emails are the same.

### Sentiment Analysis Training

We use the classic movie reviews dataset to train our Naive Bayes classifier, and apply this model to the emails. We can then see if there's a cross-correlation with positive sentiment words and spam words, etc, or if sentiment can be used at all to reliably predict spam/ham (i.e. by classifying all emails as positive or negative, and seeing which percentage of each class are spam and ham).

https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/data

### phishingcorpus

**IMPORTANT**: this file is not included in the git repository as it is quarantined by windows on download, it's available in my [fork](https://github.com/Twigonometry/MachineLearningPhishing) in case the original goes down. If you do grab the file on a Windows machine, you can add an exemption to Defender. Or you can download the file on a Linux VM. Once downloaded, put it into the `phishingcorpus-dataset` folder.

To download on Ubuntu:

```bash
$ cd phishingcorpus-dataset
$ wget https://github.com/Twigonometry/MachineLearningPhishing/tree/master/code/resources/emails-phishing.mbox
```

The "phishingcorpus" dataset from Fette et. al's paper 'Learning to Detect Phishing Emails' has been reproduced here:

https://github.com/diegoocampoh/MachineLearningPhishing/blob/master/code/resources/emails-phishing.mbox

It's quite old, so the sophistication of phishing has undoubtedly advanced since then, but it's a start.

### ML Initiation Rites

I made the mistake of committing the Enron dataset (`9ea07372587224b6481794bdd8287470038e2a83`) which is above Github's file limit. If you're faced with this lovely message

```bash
remote: error: GH001: Large files detected. You may want to try Git Large File Storage
```

Don't do what I did and make ANOTHER commit removing the file. Simply run the following:

```bash
$ git rm --cached .\kaggle-datasets\The-Enron-Email-Dataset\emails.csv
$ git commit --amend --allow-empty -C HEAD
```

(you must use `--allow-empty` if adding the dataset was the only change in the commit, but can omit the flag otherwise)

You will have to manually download, unzip, and place the Enron dataset in its corresponding folder (`kaggle-datasets/The-Enron-Email-Dataset`)
