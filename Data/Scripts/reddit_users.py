#!/usr/bin/env python3

# Search keywords similar to Twitter (she/her, he/him, they/them)
# Get comment title and author
# Exclude posts containing pronouns other than query
# Get as many posts and comments from user as possible

import csv
import praw
from praw.models import MoreComments

def credentials():
	"""Reads consumer keys and access tokens from reddit_keys.txt file"""
	with open('reddit_keys.txt', 'r') as f:
		lines = f.readlines()

	r = praw.Reddit(client_id=lines[0].split('=')[1].strip(),
					client_secret=lines[1].split('=')[1].strip(),
					user_agent='PyDisc Extraction: v1.2.5')

	return r


def main():
	r = credentials()

	with open('../')
	for comment in r.subreddit('all').search(' she/her ', limit=500):
		print(comment.title, comment.author)

if __name__ == '__main__':
	main()