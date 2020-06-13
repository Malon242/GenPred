#!/usr/bin/env python3

# This program splits the cleaned Reddit data into train/validation/test sets by
# dividing the dataframes based on usernames, dividing those groups into even chunks
# of a number of rows and combining the submissions in those rows. From this, new dataframes
# are created with rows consisting of username, list of submissions, and gender. These data-
# frames are then split up into train/validation/test sets.
# Currently limited to chunks of about 4 rows and train/validation/test of 70/15/15 

import numpy as np
import pandas as pd

def read_data():
	"""Read csv files and add column with gender"""
	nb = pd.read_csv('../Reddit Data/r_clean_nb.csv')
	nb['gender'] = 'NB'
	m = pd.read_csv('../Reddit Data/r_clean_m.csv')
	m['gender'] = 'M'
	f = pd.read_csv('../Reddit Data/r_clean_f.csv')
	f['gender'] = 'F'

	return [nb, m, f]

def combine(df):
	"""Divide dataframe into groups based on username. Divide those groups into even
	chunks of 4 rows (remainder is added to the first few chunks by 1). Combine submissions
	per chunk in a list and create new dataframe of usernames and combined submissions"""
	header = ['username', 'submissions', 'gender']
	df_list = []

	df_grouped = df.groupby('username')
	for name, group in df_grouped:
		nr_chunks = len(group)//4
		group = group.sample(frac=1, random_state=1) # Randomize row order
		
		for chunk in np.array_split(group, nr_chunks):
			submissions = chunk['submission'].tolist()
			df_list.append([name, submissions, df['gender'][0]])

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
		train, validation, test = np.split(df_new.sample(frac=1, random_state=1), [int(.7*len(df_new)), int(.85*len(df_new))])
		train_list.append(train)
		validation_list.append(validation)
		test_list.append(test)
		
	# Concatenate dataframes in list and shuffle order of rows
	train_df = pd.concat(train_list, ignore_index=True).sample(frac=1, random_state=1)
	validation_df = pd.concat(validation_list, ignore_index=True).sample(frac=1, random_state=1)
	test_df = pd.concat(test_list, ignore_index=True).sample(frac=1, random_state=1)

	# Write dataframes to csv
	train_df.to_csv('../Final Data/r_train.csv', index=False)
	validation_df.to_csv('../Final Data/r_validation.csv', index=False)
	test_df.to_csv('../Final Data/r_test.csv', index=False)

if __name__ == '__main__':
	main()
