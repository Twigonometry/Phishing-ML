# Imports
import pandas as pd
import pymongo

myclient = pymongo.MongoClient("mongodb://127.0.0.1:27017/my_database")
mydb = myclient["mydatabase"]
mycol = mydb["emails"]

testitem = {
    "body": "Test Email",
    "ratings": {
        "authoritative": 1,
        "threatening": 1,
        "rewarding": 1,
        "unnatural": 1,
        "emotional": 1,
        "provoking": 1,
        "timesensitive": 1,
        "imperative": 1
    }
  }

x = mycol.insert_one(testitem)