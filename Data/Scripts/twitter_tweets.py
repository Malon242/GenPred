#!/usr/bin/env python3

import csv
import pandas as pd

def read_data():
	"""Read csv files containing usernames and status counts"""
	nb = pd.read_csv("../Twitter Data/tw_users_nb.csv", names=['username', 'status_count'])
	m = pd.read_csv("../Twitter Data/tw_users_m.csv", names=['username', 'status_count'])
	f = pd.read_csv("../Twitter Data/tw_users_f.csv", names=['username', 'status_count'])
	
	return nb, m, f


def filter_users(lst):
	"""Remove duplicate users from dataframes,  
	filter out users with less than 1000 statuses,
	and reduce dataframes to 300 users per frame"""
	new_lst = []

	for frame in lst:
		frame = frame.drop_duplicates()
		frame = frame[frame['status_count'] > 1000]
		frame = frame.drop(frame.index[299:-1])
		new_lst.append(frame)

	return new_lst


def get_tweets(df, file):
	"""Get 250 per user, except retweets and replies, and write to file"""
	filename = '../Twitter Data/{}'.format(file)
	header = ['id', 'username', 'tweet']
	with open(filename, 'a+') as f:
		csv_writer = csv.writer(f)
		csv_writer.writerow(header)
		print("OK")
#		for user in tweepy.Cursor(api.search_users, "they/them").items(1000):
#			nb.write('{}, {}\n'.format(user.screen_name, user.statuses_count))
# id, screen_name, tweet.text



def main():
	users_nb, users_m, users_f = read_data()

	raw_frames = [users_nb, users_m, users_f]
	filtered_frames = filter_users(raw_frames)

	get_tweets(filtered_frames[0], 'tw_tweets_nb.csv') # NON-BINARY




if __name__ == '__main__':
	main()