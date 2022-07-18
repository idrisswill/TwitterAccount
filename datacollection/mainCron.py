#!/usr/bin/env python
from ClassUser import User, send_request
import pandas as pd
import os
import sys
import time
import random
from dotenv import load_dotenv
from tqdm import tqdm
from Tweets import Tweets
from responseTweets import ResponseTweets
from pymongo import MongoClient
from datetime import date
import datetime

errors = []
load_dotenv()
data_dir = "/data/AllProjectIA/TwitterAccount/datacollection"
try:
    client = MongoClient("mongodb+srv://" + os.getenv("MONGODB_USERNAME") + \
                         ":" + os.getenv("MONGODB_PASSWORD") + \
                         "@" + os.getenv("MONGODB_DOMAIN") + \
                         "/" + os.getenv("MONGODB_DBNAME") + "?retryWrites=true&w=majority")
    client.server_info()
except Exception as e:
    with open('readme.txt', 'w') as f:
        f.write(str(e))
        f.write('\n')
        f.close()
    script = "./mail_report.sh \"ERROR CONNECTION TO DATABASES\""
    os.system(script)
    sys.exit(0)

headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:78.0) Gecko/20100101 Firefox/78.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
    'Authorization': f'Bearer {os.getenv("TOKEN")}'
}

# creation du journal et redirection de la sortie standard d'erreur
orig_stdout = sys.stdout
day = date.today().strftime('%Y-%m-%dT%H:%M')
file = f'{data_dir}/journal/TaksAs-{day}.txt'
f = open(file, 'w')
sys.stdout = f
print(f"*********** start {day} ****************\n\n")
print(f"       ******* Actualize Tweet and followers***********     \n")
dict_obj_politician = {}
list_politician = ['KamtoOfficiel', 'CabralLibii', 'Prof_Meon', 'JoshuaOsih',
                   'nouranefoster', 'S_EspoirMatomba', 'OwonaGregoire',
                   'KaniBanda', 'MRC_CRM_237', 'hon_issi',
                   ]

print("\n OVERRIDE\n")

for name in list_politician:
    obj_name = User(username=name)
    obj_name.override()
    dict_obj_politician[name] = obj_name

print("\n POLITICIANS OVERRIDE\n")
dict_obj_politician

for name in list_politician:
    print(f'Start actualisation for politician {name}')
    dict_obj_politician[name].is_all_data = False
    dict_obj_politician[name].save_all_tweets(client, what_i_do='MAJ')
    dict_obj_politician[name].is_all_data = False
    dict_obj_politician[name].save_all_followers(client, what_i_do='MAJ')
    print(f'is ok for politician {name}\n\n')

print(f"       ******* Get discussion of last 7 days ***********     \n")

tweets = client["TwetterAnalitics"]["tweets"].find({
    "created_at": {
        "$gte": (date.today() - datetime.timedelta(days=6)).strftime('%Y-%m-%dT%H:%M:%S.%f%z'),
        "$lt": date.today().strftime('%Y-%m-%dT%H:%M:%S.%f%z')
    }
})
LIST_TWEETS = list(tweets)
tweets_df = pd.DataFrame(LIST_TWEETS)
LIST_CONVERSATION_ID = tweets_df['conversation_id']
LIST_CONVERSATION_ID = list(dict.fromkeys(LIST_CONVERSATION_ID))
i = 1
for conversation_id in tqdm(LIST_CONVERSATION_ID):
    print(f'conversation_id: {conversation_id}\n')
    url = f'https://api.twitter.com/2/tweets/search/recent?query=conversation_id:{conversation_id}' \
          f'&tweet.fields=author_id,conversation_id,created_at,in_reply_to_user_id,referenced_tweets' \
          f'&expansions=author_id,in_reply_to_user_id,referenced_tweets.id&user.fields=name,username' \
          f'&max_results=100'
    response = send_request(url=url, headers=headers)
    response_tweet = ResponseTweets(data=response, conversation_id=conversation_id)
    dictionary = vars(response_tweet)
    if bool(dictionary):
        client["TwetterAnalitics"]["conversations"].update_many({"created_at": dictionary["created_at"]},
                                                            {"$set": dictionary},
                                                            upsert=True)
    print(f'save number {i}\n')
    print(f'the {conversation_id} is succeed \n')
    i+=1
i = 0
print(f"       ******* send email ***********     \n")
script = f"./mail_report.sh \"ERROR CONNECTION TO DATABASES\" {file}"
os.system(script)
sys.stdout = orig_stdout
