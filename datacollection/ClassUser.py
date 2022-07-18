import random
import time
import requests
import os
from dotenv import load_dotenv
from tqdm import tqdm
from Tweets import Tweets
from Followers import Followers
from IPython.display import display, Markdown
load_dotenv()
header = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:78.0) Gecko/20100101 Firefox/78.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
    'Authorization': f'Bearer {os.getenv("TOKEN")}'
}


def send_request(url, headers):
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        print(f'OK for url:  {url}')
        result = r.json()
        return result
    else:
        print(f'can not revolve url: {url} ')
        print(f'error can be:\n')
        print(f'error http:{r.raise_for_status()}')
        return


class User:

    def __init__(self, username):
        self.headers = header
        self.url_base = 'https://api.twitter.com/2'
        self.url = f'{self.url_base}/users/by/username/{username}'
        self.username = username
        self.is_all_data = False
        self.followers = []
        self.tweets = []
        self.tweets_fields = ['author_id', 'conversation_id', 'created_at', 'in_reply_to_user_id', 'lang',
                              'context_annotations', 'public_metrics', 'possibly_sensitive', 'reply_settings',
                              'referenced_tweets'
                              ]
        self.users_fields = ['id', 'name', 'username', 'public_metrics', 'created_at', 'location', 'profile_image_url',
                             'verified', 'protected', 'url']

    def override(self):
        liste = ['id', 'name', 'username', 'public_metrics', 'created_at', 'location', 'profile_image_url', 'protected',
                 'verified', 'url']
        data = self.get_infos_with_params(liste)
        self.id = data['data']['id']
        self.name = data['data']['name']
        self.followers_count = data['data']['public_metrics']['followers_count']
        self.following_count = data['data']['public_metrics']['following_count']
        self.tweet_count = data['data']['public_metrics']['tweet_count']
        self.listed_count = data['data']['public_metrics']['listed_count']
        self.created_at = data['data']['created_at']
        self.location = data['data']['location']
        self.protected = data['data']['protected']
        self.profile_image_url = data['data']['profile_image_url']
        self.verified = data['data']['verified']
        self.url = data['data']['url']

    def get_default_infos(self):
        data = send_request(url=self.url, headers=self.headers)
        return data

    def get_infos_with_params(self, liste):
        params = ','.join(liste)
        url = f'https://api.twitter.com/2/users/by/username/{self.username}?user.fields={params}'
        data = send_request(url=url, headers=self.headers)
        return data

    def retrieve_all_tweets(self, is_all=False, pagination_token=None, max_results=20):
        params = ','.join(self.tweets_fields)
        url = ''
        if not is_all:
            if pagination_token is None:
                url = f'{self.url_base}/users/{self.id}/tweets?tweet.fields={params}&max_results={max_results}'
            else:
                url = f'{self.url_base}/users/{self.id}/tweets?tweet.fields={params}' \
                      f'&max_results={max_results}&pagination_token={pagination_token}'
            data = send_request(url=url, headers=self.headers)
            meta = data['meta']
            if 'next_token' in meta:
                return data, meta['next_token']
            else:
                return data, None

    def retrieve_all_users(self, pagination_token=None, max_results=20):
        params = ','.join(self.users_fields)
        url = ''
        if pagination_token is None:
            url = f'{self.url_base}/users/{self.id}/followers?user.fields={params}&max_results={max_results}'
        else:
            url = f'{self.url_base}/users/{self.id}/followers?user.fields={params}' \
                  f'&max_results={max_results}&pagination_token={pagination_token}'
        data = send_request(url=url, headers=self.headers)
        meta = data['meta']
        if 'next_token' in meta:
            return data, meta['next_token']
        else:
            return data, None

    def generator(self):
        while not self.is_all_data:
            yield

    def save_all_tweets(self, mongoclient, what_i_do='MAJ'):
        token = None
        for _ in tqdm(self.generator()):
            data, token = self.retrieve_all_tweets(max_results=100, pagination_token=token)
            data = data['data']
            for value in data:
                tweet = Tweets(value)
                dictionary = vars(tweet)
                mongoclient["TwetterAnalitics"]["tweets"].update_many({"created_at": dictionary["created_at"]},
                                                                      {"$set": dictionary},
                                                                      upsert=True)
            if token is None:
                self.is_all_data = True
            if what_i_do == "MAJ":
                self.is_all_data = True
                print("mise a jour terminé")

    def save_all_followers(self, mongoclient, what_i_do='MAJ'):
        token = None
        for _ in tqdm(self.generator()):
            data, token = self.retrieve_all_users(max_results=100, pagination_token=token)
            data = data['data']
            doc = self.username if self.username != 'CabralLibii' else 'followers'
            for value in data:
                follower = Followers(value)
                dictionary = vars(follower)
                mongoclient["TwetterAnalitics"][doc].update_many({"created_at": dictionary["created_at"]},
                                                                 {"$set": dictionary},
                                                                 upsert=True)
            if token is None:
                self.is_all_data = True
            if what_i_do == "MAJ":
                self.is_all_data = True
                print("mise a jour terminé")
            s = random.randint(45, 60)
            time.sleep(s)

    @classmethod
    def __str__(self):
        '''Basic whoami method'''
        display(Markdown(f'<br>**information about - {self.username}**'))
        print('Id                   :', self.id)
        print('Name                 :', self.name)
        print('number of tweet      :', self.tweet_count)
        print('number of following  :', self.following_count)
        print('number of followers  :', self.followers_count)
