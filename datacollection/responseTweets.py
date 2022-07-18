from ClassUser import send_request
from ClassUser import header


class ResponseTweets:
    def __init__(self, data, conversation_id):
        if 'data' in data:
            self.conversation_id = conversation_id
            self.list_user_and_response = []
            self.number_of_node = len(data['includes']['tweets'])
            self.token = ''
            self.text = ''
            self.created_at = ''
            self.author = ''
            self.override_data(data=data, conversation_id=conversation_id)

    def override_data(self, data, conversation_id):
        for element_data in data['data']:
            if element_data['referenced_tweets'][0]['id'] == conversation_id:
                dictionary = {'author_id': element_data['author_id'], 'created_at': element_data['created_at'],
                              'text': element_data['text']}
                self.list_user_and_response.append(dictionary)

        for include in data['includes']['tweets']:
            if include['conversation_id'] == conversation_id:
                self.text = include['text']
                self.created_at = include['created_at']
                self.author = include['author_id']
                break

        if 'next_token' in data['meta']:
            self.token = data['meta']['next_token']
        else:
            self.token = ''

        if self.token != '':
            url = f'https://api.twitter.com/2/tweets/search/recent?query=conversation_id:{conversation_id}' \
                  f'&tweet.fields=author_id,conversation_id,created_at,in_reply_to_user_id,referenced_tweets' \
                  f'&expansions=author_id,in_reply_to_user_id,referenced_tweets.id' \
                  f'&user.fields=name,username&next_token={self.token}&max_results=100'
            response = send_request(url=url, headers=header)
            self.override_data(data=response, conversation_id=conversation_id)

    def __str__(self):
        return f'id: {self.conversation_id} \ntext: {self.text}, \nnumber of nodes: {self.number_of_node}'
