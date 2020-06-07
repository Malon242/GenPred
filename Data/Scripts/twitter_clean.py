#!/usr/bin/env python3

import re
import pandas as pd

def read_data():
	"""Read csv files"""
	nb = pd.read_csv('../Twitter Data/tw_tweets_nb.csv')
	m = pd.read_csv('../Twitter Data/tw_tweets_m.csv')
	f = pd.read_csv('../Twitter Data/tw_tweets_f.csv')
	

	return nb, m, f


def main():
	nb, m, f = read_data()



	#print(nb.shape, m.shape, f.shape)
	#print(nb['username'].nunique(), m['username'].nunique(), f['username'].nunique())

if __name__ == '__main__':
	main()