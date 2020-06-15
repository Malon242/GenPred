#!/usr/bin/env python3

'''
Explanation...
'''

import re
import pandas as pd
from ast import literal_eval

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def read_data():
	"""Read csv files"""
	train = pd.read_csv('../Data/Final Data/tw_train.csv', converters={'tweets':literal_eval})
	val = pd.read_csv('../Data/Final Data/tw_validation.csv', converters={'tweets':literal_eval})
	test = pd.read_csv('../Data/Final Data/tw_test.csv', converters={'tweets':literal_eval})
	
	return [train, val, test]


def main():
	data = read_data()
	
	# Dummy classifier as baseline
	dum = DummyClassifier(strategy='most_frequent', random_state=1)

	for df in data:
		df['label'] = df['gender'].apply(lambda x: 1 if x == 'NB' else 0)

	# Baseline accuracy and classification report
	dum.fit(data[0]['tweets'], data[0]['label'])
	pred = dum.predict(data[1]['tweets'])
	print("Accuracy score: {}".format(accuracy_score(data[1]['label'], pred)))
	print("Classification report:")
	print(classification_report(data[1]['label'], pred))


if __name__ == '__main__':
	main()