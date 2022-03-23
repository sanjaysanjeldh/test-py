from typing import Optional

from fastapi import FastAPI
import tweepy # tweepy module to interact with Twitter
import pandas as pd # Pandas library to create dataframes
from tweepy import OAuthHandler # Used for authentication
from tweepy import Cursor # Used to perform pagination
import config
import uvicorn
from predict import run
from medium import get_tweets_from_user

app = FastAPI()


cons_key = config.api_key
cons_secret = config.api_key_secret
acc_token = config.access_token
acc_secret = config.access_token_secret


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get('/{id}')
def get_tweets(id:str,data:Data):
    if(id == 'usemabassy'):
        df = get_tweets_from_user("usembassynepal")
    elif(id =='finlandinnepal'):
        df = get_tweets_from_user("finlandinnepal")
        print("File written to a CSV FIle")
    elif(id == 'pakinnepal'):
        df = get_tweets_from_user("pakinnepal")
        print("File written to a CSV FIle")
    

    return run(df,id)

if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8001)