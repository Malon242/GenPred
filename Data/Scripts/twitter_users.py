#!/usr/bin/env python3

'''
This program searches for non-binary, male, and female twitter users, who have 
explicitely stated their pronouns in their username. The usernames and status count
are written to a textfile corresponding to the selfproclaimed gender of the user.
Currently limited to a 1000 usernames per gender, duplicates possible
'''

import tweepy

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


def main():
	"""Search for users with query term in twitter username,
	write username and status count to textfile"""

	api = credentials()

	# NON-BINARY
	with open('../Twitter Data/tw_users_nb.txt', 'a+') as nb:
		for user in tweepy.Cursor(api.search_users, "they/them").items(1000):
			nb.write('{}, {}\n'.format(user.screen_name, user.statuses_count))

	# MALE
	with open('../Twitter Data/tw_users_m.txt', 'a+') as m:
		for user in tweepy.Cursor(api.search_users, "he/him").items(1000):
			m.write('{}, {}\n'.format(user.screen_name, user.statuses_count))

	# FEMALE
	with open('../Twitter Data/tw_users_f.txt', 'a+') as f:
		for user in tweepy.Cursor(api.search_users, "she/her").items(1000):
			f.write('{}, {}\n'.format(user.screen_name, user.statuses_count))


if __name__ == '__main__':
	main()