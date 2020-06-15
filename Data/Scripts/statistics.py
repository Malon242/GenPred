#!/usr/bin/env python3

'''
This program calculates and prints statistics for the Twitter and Reddit
training, validation and test sets. The statistics contain information about 
the number of rows (size) of the data sets, the number of unique users per gender
and the average length of the combined posts per gender and in total. 
'''

import pandas as pd

def read_data():
	"""Read csv files and add name to dataframe"""
	tw_train = pd.read_csv('../Final Data/tw_train.csv')
	tw_train.name = 'Twitter training data'
	tw_val = pd.read_csv('../Final Data/tw_validation.csv')
	tw_val.name = 'Twitter validation data'
	tw_test = pd.read_csv('../Final Data/tw_test.csv')
	tw_test.name = 'Twitter test data'
	r_train = pd.read_csv('../Final Data/r_train.csv')
	r_train.name = 'Reddit training data'
	r_val = pd.read_csv('../Final Data/r_validation.csv')
	r_val.name = 'Reddit validation data'
	r_test = pd.read_csv('../Final Data/r_test.csv')
	r_test.name = 'Reddit test data'

	return [tw_train, tw_val, tw_test, r_train, r_val, r_test]

def main():

	data = read_data()

	for df in data:
		print("\n----------\nDATAFRAME STATISTICS: {}".format(df.name))
		
		# Rows gender
		print('Number of rows per gender:')
		print(df.groupby('gender').count())
		# Total rows
		print('Total rows: {}'.format(df.shape[0]))
		
		# Users gender
		print('\nNumber of users per gender:')
		print(df.groupby('gender')['username'].nunique())
		# Total users
		print('Total users: {}'.format(df['username'].nunique()))

		# Average length posts gender
		print('\nAverage length of posts per gender: ')
		grouped = df.groupby('gender')
		for name, group in grouped:
			print(name, group[group.columns[1]].apply(len).mean())
		# Average length total posts
		print('Average length total posts: {}'.format(df[df.columns[1]].apply(len).mean()))
		
		print('----------')

if __name__ == '__main__':
	main()
