#!/usr/bin/env python3

# This program splits the cleaned Twitter data into train/validation/test sets by
# dividing the dataframes based on usernames, dividing those groups into even chunks
# of a number of rows and combining the tweets in those rows. From this new dataframes
# are created with rows consisting of username, list of tweets, and gender. These data-
# frames are then split up into train/validation/test sets.
# Currently limited to chunks of about 5 rows and train/validation/test of 60/20/20 

import numpy as np
import pandas as pd

def read_data():
	"""Read csv files and add column with gender"""
	nb = pd.read_csv('../Twitter Data/tw_clean_nb.csv')
	nb['gender'] = 'NB'
	m = pd.read_csv('../Twitter Data/tw_clean_m.csv')
	m['gender'] = 'M'
	f = pd.read_csv('../Twitter Data/tw_clean_f.csv')
	f['gender'] = 'F'

	return [nb, m, f]


def combine(df):
	"""Divide dataframe into groups based on username. Divide those groups into even
	chunks of 5 rows (remainder is added to the first few chunks by 1). Combine tweets
	per chunk in a list and create new dataframe of usernames and combined tweets"""
	header = ['username', 'tweets', 'gender']
	df_list = []

	df_grouped = df.groupby('username')
	for name, group in df_grouped:
		nr_chunks = len(group)//5 
		group = group.sample(frac=1, random_state=1) # Randomize row order
		
		for chunk in np.array_split(group, nr_chunks):
			tweets = chunk['tweet'].tolist()
			df_list.append([name, tweets, df['gender'][0]])

	df_new = pd.DataFrame(df_list, columns=header)
	return df_new


def main():
	data = read_data()
	np.random.seed(1)

	train_list = []
	validation_list = []
	test_list = []

	for df in data:
		df_new = combine(df)

		# Split dataframe into train/validation/test sets and append to list
		train, validation, test = np.split(df_new.sample(frac=1, random_state=1), [int(.6*len(df_new)), int(.8*len(df_new))])
		train_list.append(train)
		validation_list.append(validation)
		test_list.append(test)
		
	# Concatenate dataframes in list and shuffle order of rows
	train_df = pd.concat(train_list, ignore_index=True).sample(frac=1, random_state=1)
	validation_df = pd.concat(validation_list, ignore_index=True).sample(frac=1, random_state=1)
	test_df = pd.concat(test_list, ignore_index=True).sample(frac=1, random_state=1)

	# Write dataframes to csv
	train_df.to_csv('../Final Data/tw_train.csv', index=False)
	validation_df.to_csv('../Final Data/tw_validation.csv', index=False)
	test_df.to_csv('../Final Data/tw_test.csv', index=False)

if __name__ == '__main__':
	main()
