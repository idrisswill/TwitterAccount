
class Followers:
    def __init__(self, data):
        self.id = data['id']
        self.name = data['name']
        self.username = data['username']
        self.created_at = data['created_at']
        self.followers_count = data['public_metrics']['followers_count']
        self.following_count = data['public_metrics']['following_count']
        self.tweet_count = data['public_metrics']['tweet_count']
        self.listed_count = data['public_metrics']['listed_count']
        self.protected = data['protected']
        self.profile_image_url = data['profile_image_url']
        self.verified = data['verified']
        self.url = data['url']
        if 'location' in data:
            self.location = data['location']

    def __str__(self):
        return f'id: {self.id} name: {self.name}, followers: {self.followers_count},'



