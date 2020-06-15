#!/usr/bin/env python3

'''
This program uses csv files containing Reddit usernames. It first removes 
duplicates from these files, and reduces the number of users. For every 
remaining user it then scrapes posts and comments and writes these to a csv file.
Currently limited to 250 users per gender, maximum 150 comments and 100 posts per user.
'''

import csv
import praw
import pandas as pd

def credentials():
	"""Reads consumer keys and access tokens from reddit_keys.txt file"""
	with open('reddit_keys.txt', 'r') as f:
		lines = f.readlines()

	r = praw.Reddit(client_id=lines[0].split('=')[1].strip(),
					client_secret=lines[1].split('=')[1].strip(),
					user_agent='PyUser Scraper: v1.5.1 (by /u/Maves23)')

	return r


def read_data():
	"""Read first column csv files containing usernames"""
	nb = pd.read_csv("../Reddit Data/r_users_nb.csv", usecols=[0])
	m = pd.read_csv("../Reddit Data/r_users_m.csv", usecols=[0])
	f = pd.read_csv("../Reddit Data/r_users_f.csv", usecols=[0])
	
	return nb, m, f


def filter_users(lst, r):
	"""Remove duplicate users from dataframes, and 
	reduce dataframes to 250 users per frame"""
	new_lst = []

	for frame in lst:
		frame = frame.drop_duplicates()
		frame = frame.sample(n=250, random_state=1).reset_index(drop=True)
		new_lst.append(frame)

	return new_lst


def submission_scrape(csv_writer, row, r):
	"""Scrape both posts+title and comments from a redditor
	Write username and texts to a csv file"""
	print(row['author'])

	try:
		for comment in r.redditor(row['author']).comments.new(limit=150):
			csv_writer.writerow([row['author'], None, comment.body])
		for submission in r.redditor(row['author']).submissions.new(limit=100):
			csv_writer.writerow([row['author'], submission.title, submission.selftext])
	except Exception as e:
		print(e)


def get_submissions(df, file, r):
	"""Create a csv file and scrape data for every user in the dataframe"""
	filename = '../Reddit Data/{}'.format(file)
	header = ['username', 'title', 'text']

	with open(filename, 'a+') as f:
		csv_writer = csv.writer(f)
		csv_writer.writerow(header)
		df.apply(lambda row: submission_scrape(csv_writer, row, r), axis=1)


def main():
	r = credentials()
	users_nb, users_m, users_f = read_data()

	raw_frames = [users_nb, users_m, users_f]
	filtered_frames = filter_users(raw_frames, r)

	get_submissions(filtered_frames[0], 'r_submissions_nb.csv', r) # NON-BINARY
	get_submissions(filtered_frames[1], 'r_submissions_m.csv', r) # MALE
	get_submissions(filtered_frames[2], 'r_submissions_f.csv', r)  # FEMALE

if __name__ == '__main__':
	main()
