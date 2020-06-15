#!/usr/bin/env python3

'''
This program searches for Reddit submissions which include the gender keywords provided
in the query. It then writes the author's name, submission title, and submission text
to a csv file. 
Currently limited to at most 500 submissions per gender, not filtered, duplicate possible
'''

import csv
import praw

def credentials():
	"""Reads consumer keys and access tokens from reddit_keys.txt file"""
	with open('reddit_keys.txt', 'r') as f:
		lines = f.readlines()

	r = praw.Reddit(client_id=lines[0].split('=')[1].strip(),
					client_secret=lines[1].split('=')[1].strip(),
					user_agent='PyUser Extraction: v1.0')

	return r


def submissions(file, term, r, lim_nr):
	"""Create csv file and add author, submission title and text 
	from search based on gender keywords."""
	filename = '../Reddit Data/{}'.format(file)
	header = ['author', 'title', 'text']

	with open(filename, 'a+') as f:
		csv_writer = csv.writer(f)
		csv_writer.writerow(header)

		for submission in r.subreddit('all').search(term, limit=lim_nr):
			csv_writer.writerow([submission.author, submission.title, submission.selftext])
			print(submission.author)


def main():
	r = credentials()

	submissions('r_users_nb.csv', 'they/them', r, 500) # NON-BINARY
	submissions('r_users_nb.csv', 'I am non-binary', r, 200) #NON-BINARY
	submissions('r_users_m.csv', 'he/him', r, 500) # MALE
	submissions('r_users_m.csv', 'I am a man', r, 200) # MALE
	submissions('r_users_f.csv', 'she/her', r, 500) # FEMALE
	submissions('r_users_f.csv', 'I am a woman', r, 200) # FEMALE

if __name__ == '__main__':
	main()