#!/usr/bin/env python3

# This program uses csv files containing Twitter usernames and number of statuses.
# It first removes duplicates from these files, and filters out users based on 
# the number of statuses and the total number of users wanted. For every remaining 
# user it then scrapes tweets from the timeline (excluding retweets and replies) 
# and writes these to a csv file.
# Currently limited to 200 users per gender and up to 200 tweets per user.

import csv
import time
import tweepy
import pandas as pd

def credentials():
	"""Reads consumer keys and access tokens from twitter_keys.txt file"""
	with open('twitter_keys.txt', 'r') as f:
		lines = f.readlines()

	auth = tweepy.OAuthHandler(lines[0].split('=')[1].strip(), 
		lines[1].split('=')[1].strip())
	auth.set_access_token(lines[2].split('=')[1].strip(), 
		lines[3].split('=')[1].strip())

	api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

	return api


def read_data():
	"""Read csv files containing usernames and status counts"""
	nb = pd.read_csv("../Twitter Data/tw_users_nb.csv", names=['username', 'status_count'])
	m = pd.read_csv("../Twitter Data/tw_users_m.csv", names=['username', 'status_count'])
	f = pd.read_csv("../Twitter Data/tw_users_f.csv", names=['username', 'status_count'])
	
	return nb, m, f


def filter_users(lst):
	"""Remove duplicate users from dataframes, filter out users with 
	less than 1000 statuses, and reduce dataframes to 200 users per frame"""
	new_lst = []

	for frame in lst:
		frame = frame.drop_duplicates()
		frame = frame[frame['status_count'] > 1000]
		frame = frame.drop(frame.index[199:-1]).reset_index(drop=True)
		new_lst.append(frame)

	return new_lst


def tweet_scrape(csv_writer, row, api):
	"""Scrape tweets from user's timeline, excluding retweets and replies.
	Write username and tweettext to csv file"""
	try:
		for tweet in tweepy.Cursor(api.user_timeline, screen_name=row['username'], 
			exclude_replies=True, include_rts=False).items(200):
			csv_writer.writerow([row.name, row['username'], tweet.text])
	except tweepy.TweepError as e:
		if e == "[{u'message': u'Rate limit exceeded', u'code': 88}]":
			time.sleep(60*5) #Sleep for 5 minutes
		else:
			print(e)


def get_tweets(df, file, api):
	"""Create csv file and apply tweet scraping funtion for every user in the dataframe"""
	filename = '../Twitter Data/{}'.format(file)
	header = ['id', 'username', 'tweet']
	with open(filename, 'a+') as f:
		csv_writer = csv.writer(f)
		csv_writer.writerow(header)
		df.apply(lambda row: tweet_scrape(csv_writer, row, api), axis=1)


def main():
	api = credentials()
	users_nb, users_m, users_f = read_data()

	raw_frames = [users_nb, users_m, users_f]
	filtered_frames = filter_users(raw_frames)

	get_tweets(filtered_frames[0], 'tw_tweets_nb.csv', api) # NON-BINARY
	get_tweets(filtered_frames[1], 'tw_tweets_m.csv', api) # MALE
	get_tweets(filtered_frames[2], 'tw_tweets_f.csv', api) # FEMALE


if __name__ == '__main__':
	main()
