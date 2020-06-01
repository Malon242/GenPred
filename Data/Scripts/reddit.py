#!/usr/bin/env python3

import praw
from praw.models import MoreComments
import client

def main():
	reddit = praw.Reddit(client_id=,
					client_secret=,
					user_agent=)

	for comment in reddit.subreddit('ftm').top('year'):
		print(comment.author)

if __name__ == '__main__':
	main()