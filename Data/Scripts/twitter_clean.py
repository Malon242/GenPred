#!/usr/bin/env python3

'''
This program cleans the original Twitter data by adding tags, removing a number of 
automatically generated tweets, and removing duplicate tweets.
It also removes short tweets and users with a small number of tweets.
Currently the length of a tweet is at least 10 parts and 
a user has to have at least 10 tweets
'''

import re
import pandas as pd

def read_data():
	"""Read csv files"""
	nb = pd.read_csv('../Twitter Data/tw_tweets_nb.csv').dropna()
	m = pd.read_csv('../Twitter Data/tw_tweets_m.csv').dropna()
	f = pd.read_csv('../Twitter Data/tw_tweets_f.csv').dropna()

	return [nb, m, f]


def clean_tweet(tweet):
	"""Clean tweets by replacing links and @'s with tags 
	and removing common automatically generated tweets"""
	tweet = re.sub(r"\bhttps?:\/\/\S+\b", '<LINK>', tweet) # Replace links with <LINK> tag
	tweet = re.sub(r"@\w+", '<USER> ', tweet) # Replace @user with <USER> tag
	tweet = re.sub(r"😺✏ — ((?s).*?)<LINK>", r"<CATASK> \1", tweet) # Add a tag to CuriousCat answers
	tweet = re.sub(r"\[ID(.*?)\]", '<DESCRIPTION>', tweet, flags=re.I) 
	tweet = re.sub(r"\[alt(.*?)\]", '<DESCRIPTION>', tweet, flags=re.I)
	tweet = re.sub(r"\[desc(.*?)\]", '<DESCRIPTION>', tweet, flags=re.I)

	# Replace automatically generated text and short tweets with None
	to_be_removed = ['My week on Twitter', 'My fitbit #Fitstats', 'biggest fans this week',
	'via @YouTube', 'automatically checked by', '#MyTwitterAnniversary']
	if any(n in tweet for n in to_be_removed) or len(tweet.split(' '))<10: 
		tweet = None

	return tweet


def main():
	data = read_data()
	file_names = ['../Twitter Data/tw_clean_nb.csv', 
				  '../Twitter Data/tw_clean_m.csv', 
				  '../Twitter Data/tw_clean_f.csv']

	for df, f in zip(data, file_names):
		df['tweet'] = df['tweet'].apply(lambda x: clean_tweet(x))
		df = df.dropna().drop_duplicates()
		df = df.groupby('username').filter(lambda x: len(x)>10)
		df.to_csv(f, index=False)

if __name__ == '__main__':
	main()
