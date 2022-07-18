from TwitterAccount.datacollection.ClassUser import User
import os
from dotenv import load_dotenv
from pymongo import MongoClient



load_dotenv()
try:
    client = MongoClient("mongodb+srv://" + os.getenv("MONGODB_USERNAME") + \
                         ":" + os.getenv("MONGODB_PASSWORD") + \
                         "@" + os.getenv("MONGODB_DOMAIN") + \
                         "/"+os.getenv("MONGODB_DBNAME")+"?retryWrites=true&w=majority")
    client.server_info()
except Exception as e:
    raise e


user = User(username='KamtoOfficiel')
user.override()
print(user.username)
print('followers',user.followers_count)
print(user.tweet_count)
user.save_all_followers(client)