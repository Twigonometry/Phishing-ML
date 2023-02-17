# Phishing-ML
Experimental repository on phishing datasets for dissertation

## Datasets

Email-Spam-Dataset: https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset
- License: https://creativecommons.org/publicdomain/zero/1.0/

Phishing-Dataset-for-Machine-Learning: https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning
- License: https://creativecommons.org/licenses/by/4.0/

The-Enron-Email-Dataset: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset (must be manually placed in folder, see below)
- License: copyright original author

Web-page-Phishing-Detection-Dataset: https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset
- License: https://creativecommons.org/licenses/by/4.0/

### ML Initiation Rites

I made the mistake of committing the Enron dataset (`9ea07372587224b6481794bdd8287470038e2a83`) which is above Github's file limit. If you're faced with this lovely message

```
remote: error: GH001: Large files detected. You may want to try Git Large File Storage
```

Don't do what I did and make ANOTHER commit removing the file. Simply run the following:

```
$ git rm --cached .\kaggle-datasets\The-Enron-Email-Dataset\emails.csv
$ 
```

You will have to manually download, unzip, and place the Enron dataset in its corresponding folder (`kaggle-datasets/The-Enron-Email-Dataset`)