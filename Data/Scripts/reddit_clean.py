#!/usr/bin/env python3

'''
This program cleans the original Reddit data by adding tags, removing a number
of submissions that are too short, or have already been removed by the user, or are 
duplicate submissions. User with a small number of submissions are also removed.
Currently the length of a submission is at least 8 parts and 
a user has to have at least 10 submissions
'''

import re
import pandas as pd

def read_data():
	"""Read csv files"""
	nb = pd.read_csv('../Reddit Data/r_submissions_nb.csv').dropna(thresh=2)
	m = pd.read_csv('../Reddit Data/r_submissions_m.csv').dropna(thresh=2)
	f = pd.read_csv('../Reddit Data/r_submissions_f.csv').dropna(thresh=2)
	
	return [nb, m, f]


def clean_text(submission):
	"""Clean submissions by replacing usernames, subreddits, links and more by corresponding
	tags. Replace common Reddit markdown by tags. Remove submissions that have been removed
	by the user"""
	submission = re.sub("&#x200B;", '\n', submission)
	submission = re.sub("&nbsp;", ' ', submission)
	submission = re.sub(r"(\/|\b)r\/\w+", '<SUBREDDIT>', submission) # Replace /r/subreddit by tag
	submission = re.sub(r"(\/|\b)u\/[\w\-]+", '<USER>', submission) # Replace /u/user by tag
	submission = re.sub(r"^>.*$", '<QUOTE>', submission, flags=re.M) # Replace '>' by quote tag
	submission = re.sub(r"\[.+\]\(https?:\/\/\S+\)", '<DESCRIPTION> <LINK>', submission) # Replace links
	submission = re.sub(r"\bhttps?:\/\/\S+\b", '<LINK>', submission) 
	submission = re.sub(r"\bTL;DR:", '<SUMMARY>', submission, flags=re.I)

	# Add tags for Reddit markdown most commonly used, except if \ used before markdown
	submission = re.sub(r"(?<!\\)(___|\*\*\*)(.*?[^\\])(\1)", r'<BOLD-ITALIC> \2', submission) # ___bold-italic___ ***bold-italic***
	submission = re.sub(r"(?<!\\)(__|\*\*)(.*?[^\\])(\1)", r'<BOLD> \2', submission) # __bold__ **bold**
	submission = re.sub(r"(?<!\\)(_|\*)(.*?[^\\])(\1)", r'<ITALIC> \2', submission) # _italic_ *italic*
	submission = re.sub(r"(?<!\\)~~(.*?[^\\])~~", r'<STRIKE> \1', submission) # ~~strikethrough~~
	submission = re.sub(r"(?<!\\)>!(.*?[^\\])!<", r'<SPOILER> \1', submission) # >!spoilers!<
	submission = re.sub(r"(?<!\\)\^\((.*[^\\])\)", r'<SUPER> \1', submission) # ^(superscript)
	submission = re.sub(r"(?<!\\)\^(\w+)", r'<SUPER> \1', submission) # ^supercript
	submission = re.sub(r"(?<!\\)#{1,6}(.*)", r'<HEADING> \1', submission) # #heading

	# Replace removed submissions and short submissions with None
	to_be_removed = ['[removed]']
	if any(i in submission for i in to_be_removed) or len(submission.split(' '))<8:
		submission = None

	return submission


def main():
	data = read_data()
	file_names = ['../Reddit Data/r_clean_nb.csv', 
				  '../Reddit Data/r_clean_m.csv', 
				  '../Reddit Data/r_clean_f.csv']
	
	for df,f in zip(data, file_names):
		df['submission'] = (df['title'].fillna('') + " \n " + df['text'].fillna('')).str.strip(' \n ')
		df['submission'] = df['submission'].apply(lambda x: clean_text(x))
		df = df.filter(items=['username', 'submission'])
		df = df.dropna().drop_duplicates()
		df = df.groupby('username').filter(lambda x: len(x)>10)
		df.to_csv(f, index=False)


if __name__ == '__main__':
	main()
