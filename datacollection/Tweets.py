class Tweets:

    def __init__(self, data):
        liste = ['id', 'name', 'username', 'public_metrics', 'created_at', 'location', 'profile_image_url',
                 'verified', 'protected', 'url']
        self.id = data['id']
        self.text = data['text']
        self.conversation_id = data['conversation_id']
        self.created_at = data['created_at']
        self.reply_count = data['public_metrics']['reply_count']
        self.like_count = data['public_metrics']['like_count']
        self.retweet_count = data['public_metrics']['retweet_count']
        self.quote_count = data['public_metrics']['quote_count']
        self.reply_settings = data['reply_settings']
        self.author_id = data['author_id']
        self.possibly_sensitive = data['possibly_sensitive']
        self.lang = data['lang']
        if 'context_annotations' in data:
            self.context_annotations_domain_name =\
                data['context_annotations'][0]['domain']['name']
            self.context_annotations_entity_name = \
                data['context_annotations'][0]['entity']['name']
            self.context_annotations_entity_id = \
                data['context_annotations'][0]['entity']['id']
        if 'referenced_tweets' in data:
            self.referenced_tweets_type = data['referenced_tweets'][0]['type']
            self.referenced_tweets_id = data['referenced_tweets'][0]['id']
        if 'in_reply_to_user_id' in data:
            self.in_reply_to_user_id = data['in_reply_to_user_id']

    def __str__(self):
        return f'id: {self.id} text: {self.text}, like_count: {self.like_count} '