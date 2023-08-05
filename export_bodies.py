import pandas as pd

"""export all of the email bodies to a csv file for data labelling
using the full text rather than preprocessed, but still need to extract enron"""

# Educational institute data

dfEdu = pd.read_csv('./educational-institute-dataset/PhishingEmailData.csv', encoding="ISO-8859-1", usecols=['Email_Subject', 'Email_Content', 'Closing_Remarks'])

dfEdu['Body'] = dfEdu[dfEdu.columns[1:]].apply(
    lambda x: '\n'.join(x.dropna().astype(str)),
    axis=1
)

dfEdu.drop(columns=['Email_Subject', 'Email_Content', 'Closing_Remarks'])

# Email Spam Dataset

dfSA = pd.read_csv('./kaggle-datasets/Email-Spam-Dataset/completeSpamAssassin.csv', usecols=['Body'])
dfEnron = pd.read_csv('./kaggle-datasets/Email-Spam-Dataset/enronSpamSubset.csv', usecols=['Body']).tail(-1)
dfLing = pd.read_csv('./kaggle-datasets/Email-Spam-Dataset/lingSpam.csv', usecols=['Body'])

# Enron Full dataset
# Reading from this file requires having run process_enron() in classif.ipynb
# We also do not remove duplicates in this file

dfEnronFull = pd.read_csv('./Processed-Datasets/Enron-Bodies/emails.csv', usecols=['message']).rename({"message":"Body"})

# Merge all

frames = [dfEdu, dfSA, dfEnron, dfLing, dfEnronFull]

pd.concat(frames).to_csv("./Processed-Datasets/all-bodies.csv")