import tweepy
import configparser
import config


#Managing all the secret config files
# config = configparser.ConfigParser()
# config.read('config.ini')
 
#Authenticating our API
# api_key = config['twitter']['api_key']
# api_key_secret = config['twitter']['api_key_secret']
# access_token = config['twitter']['access_token']
# access_token_secret = config['twitter']['access_token_secret']
# bearer_token= config['twitter']['bearer_token']

client = tweepy.Client(bearer_token=config.bearer_token )


auth = tweepy.OAuthHandler(config.api_key,config.api_key_secret)
auth.set_access_token(config.access_token,config.access_token_secret)
api = tweepy.Client(auth)

query = "from:usembassynepal"
public_tweets = api.search_recent_tweets(query=query)
for tweet in public_tweets:
    print(tweet.text)