import tweepy # tweepy module to interact with Twitter
import pandas as pd # Pandas library to create dataframes
from tweepy import OAuthHandler # Used for authentication
from tweepy import Cursor # Used to perform pagination
import config
import json

"""
Twitter Authentification Credentials
"""
cons_key = config.api_key
cons_secret = config.api_key_secret
acc_token = config.access_token
acc_secret = config.access_token_secret

# (1). Athentication Function
def get_twitter_auth():
    """
    @return:
        - the authentification to Twitter
    """
    try:
        consumer_key = cons_key
        consumer_secret = cons_secret
        access_token = acc_token
        access_secret = acc_secret
        
    except KeyError:
        sys.stderr.write("Twitter Environment Variable not Set\n")
        sys.exit(1)
        
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    
    return auth

# (2). Client function to access the authentication API
def get_twitter_client():
    """
    @return:
        - the client to access the authentification API
    """
    auth = get_twitter_auth()
    client = tweepy.API(auth, wait_on_rate_limit=True)
    return client

# (3). Function creating final dataframe
def get_tweets_from_user(twitter_user_name, page_limit=16, count_tweet=200):
    """
    @params:
        - twitter_user_namez: the twitter username of a user (company, etc.)
        - page_limit: the total number of pages (max=16)
        - count_tweet: maximum number to be retrieved from a page
        
    @return
        - all the tweets from the user twitter_user_name
    """
    client = get_twitter_client()
    
    all_tweets = []
    
    for page in Cursor(client.user_timeline, include_rts = False,
                        screen_name=twitter_user_name, 
                        count=count_tweet).pages(page_limit):
        for tweet in page:
            parsed_tweet = {}
            parsed_tweet['date'] = tweet.created_at
            parsed_tweet['author'] = tweet.user.name
            parsed_tweet['text'] = tweet.text
                
            all_tweets.append(parsed_tweet)
    
    
    print("the page is ===>",page[0]._api,"Nevermind")
    with open("student.json","w") as json_file:
        json.dump(page[1]._json,json_file,indent=4)
        
    # Create dataframe 
    df = pd.DataFrame(all_tweets)
    
    # Revome duplicates if there are any
    df = df.drop_duplicates( "text" , keep='first')
    
    #Creating CSV Files for each Embassy
    if twitter_user_name == 'usembassy':
        df.to_csv('usembassy.csv')
        print("CSV of American Embassy Saved")
    elif twitter_user_name == 'finlandinnepal':
        df.to_csv('finland.csv')
        print("CSV of Finland Embassy Saved")

    
    return df

# googleAI = get_tweets_from_user("usembassynepal")
# print("Data Shape: {}".format(googleAI))