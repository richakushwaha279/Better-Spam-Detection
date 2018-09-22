import csv
import numpy as np
import random
import matplotlib.pyplot as plt 
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')

import re
from scipy import stats

T = 100
etas = [0.5, 0.3, 0.2, 0.15, 0.1, 0.01, 0.001]
runs = 100

perceptron_miss_classification_each_round_each_trial = []
winnow_miss_classification_each_round_each_trial = []

# dictionary having key as line no and value as another dictionary having key as word and value as its count in that line
bag_of_words = {}

vocab = {}

stopWords = {}
words = stopwords.words('english')
for i in words:
	stopWords[i] = 0

miss_classification_perceptron_each_round = [0.0] * T
miss_classification_winnow_each_round = [0.0] * T

def flip_class_labels(Class_labels, no_of_class_labels_to_flip):
	total_class_labels = len(Class_labels)
	# generate no_of_class_labels_to_flip numbers to flip and save them in indices
	indices = random.sample(range(0, total_class_labels - 1), no_of_class_labels_to_flip)
	for i in xrange(len(Class_labels)):
		if i in indices:
			Class_labels[i] *= -1
	return Class_labels

def read_data(fname, flag = False):
	fd = open(fname,'r')
	freader = csv.reader(fd)
	Dataset = []
	Class_labels = []
	for line in freader:
		Dataset.append((map(float, line[:-1])))
		label = int(line[-1])
		if label == 0:
			label = -1
		Class_labels.append(label)
	fd.close()	

	if flag == True:
		no_of_class_labels_to_flip = len(Class_labels) / 10
		Class_labels = flip_class_labels(Class_labels, no_of_class_labels_to_flip)
	return Dataset, Class_labels

def TrainPerceptron(Dataset, Class_labels, eta):
	global miss_classification_perceptron_each_round, T
	size_of_dataset = len(Dataset)
	no_of_features = len(Dataset[0])
	w = np.array([0.0] * no_of_features)
	index = np.random.uniform(size = T) * size_of_dataset
	index = map(int, index)
	missClassifications = 0
	t = 0
	for i in index:
		x = np.array(Dataset[i])
		y_predict = np.dot(w, x)
		y_actual = Class_labels[i]
		if y_predict * y_actual <= 0:
			w = w + eta * y_actual * x
			missClassifications += 1
		miss_classification_perceptron_each_round[t] += missClassifications

		t += 1

	return w

def TrainWinnow(Dataset, Class_labels, eta):
	global miss_classification_winnow_each_round, T
	size_of_dataset = len(Dataset)
	no_of_features = len(Dataset[0])
	w = np.array([0.0] * no_of_features)
	w += 1.0/no_of_features # normalize the weight vector

	index = np.random.uniform(size = T) * size_of_dataset
	index = map(int, index)
	t = 0
	missClassifications = 0
	for i in index:
		x = np.array(Dataset[i])
		y_predict = np.dot(w, x)
		y_actual = Class_labels[i]

		if y_predict * y_actual <= 0:
			Z = np.sum((w * np.exp(eta * y_actual * x)))
			w = w * np.exp(eta * y_actual * x)
			w = w / Z
			missClassifications += 1
		miss_classification_winnow_each_round[t] += missClassifications

		t += 1

	return w

def plot(vector1, vector2, algo, eta, y_err):
	plt.xlabel('Rounds 1 to T')
	plt.ylabel('Miss classification avgd over 100 runs')
	if algo == 'Perceptron':
		col = 'green'
	else:
		col = 'blue'
	plt.errorbar(vector1, vector2, y_err, label = algo + str(eta), color = col)
	plt.legend(loc = 'best')

def get_feature_vector(msg):
	global vocab
	feature_vector = [0] * len(vocab)
	for word, value in vocab.iteritems():
		if msg.count(word.lower()):
			feature_vector.append(msg.count(word))
		else:
			feature_vector.append(0)

	return feature_vector

def read_data2(fname, flag = False):
	global vocab

	Dataset = []
	Class_labels = []

	# build vocab
	for line in open(fname):
		label, msg = line.split('\t')
		regex = re.compile(r'[a-zA-Z0-1]+')
		msg = regex.findall(msg)
		for word in msg:
			try:
				stopWords[word]
			except:
				vocab[word.lower()] = 1

	# build the feature vectors
	t = 0
	for line in open(fname):

		label, msg = line.split('\t')
		if label == 'ham':
			label = 1
		else:
			label = -1
		feature_vector = get_feature_vector(msg)
		feature_vector.append(1)
		Dataset.append(feature_vector)
		Class_labels.append(label)
		t+=1

	if flag == True:
		no_of_class_labels_to_flip = len(Class_labels) / 10
		Class_labels = flip_class_labels(Class_labels, no_of_class_labels_to_flip)
	return Dataset, Class_labels

def main():
	global T, etas, runs, miss_classification_perceptron_each_round, miss_classification_winnow_each_round, perceptron_miss_classification_each_round_each_trial, winnow_miss_classification_each_round_each_trial

	choice = input('Enter a choice : \n\
		1)Spambase Data Collection.\n\
		2)SMS Spam Collection\n\
		3)Spambase Data Collection with 10% flipped Class labels\n\
		4)SMS Spam Collection with 10% flipped Class labels\n')
	if choice == 1:
		Dataset, Class_labels = read_data('spambase.data')
	elif choice == 2:
		Dataset, Class_labels = read_data2('SMSSpamCollection')
	elif choice == 3:
		Dataset, Class_labels = read_data('spambase.data', True)
	elif choice == 4:
		Dataset, Class_labels = read_data2('SMSSpamCollection', True)
	else:
		print 'Wrong choice! '
		exit()

	# doubling the Dataset for Winnow Algorithm
	FinalDataset = []

	for i in range(len(Dataset)):
		x1 = Dataset[i]
		x2 = [-1.0*val for val in Dataset[i]]
		FinalDataset.append(np.append(x1,x2))

	for eta in etas:
		# empty the perceptron_miss_classification_each_round_each_trial list before calling it for different eta value
		perceptron_miss_classification_each_round_each_trial = []
		# empty the winnow_miss_classification_each_round_each_trial list before calling it for different eta value
		winnow_miss_classification_each_round_each_trial = []
		for run in range(runs):
			w = TrainPerceptron(Dataset, Class_labels, eta)

			# append the miss classification in each round to the outer list containing miss classification in each round each trial
			perceptron_miss_classification_each_round_each_trial.append(miss_classification_perceptron_each_round)

			w = TrainWinnow(FinalDataset, Class_labels, eta)

			# append the miss classification in each round to the outer list containing miss classification in each round each trial
			winnow_miss_classification_each_round_each_trial.append(miss_classification_winnow_each_round)

			print 'Run : ', run

		for i in range(len(miss_classification_perceptron_each_round)):
			miss_classification_perceptron_each_round[i] = miss_classification_perceptron_each_round[i]/float(runs)
			miss_classification_winnow_each_round[i] = miss_classification_winnow_each_round[i]/float(runs)

		print 

		y_err1 = stats.sem(perceptron_miss_classification_each_round_each_trial)
		y_err2 = stats.sem(winnow_miss_classification_each_round_each_trial)

		rounds = [i for i in range(1,T+1)]
		plot(rounds, miss_classification_perceptron_each_round, 'Perceptron', eta, y_err1)
		plot(rounds, miss_classification_winnow_each_round, 'Winnow', eta, y_err2)
		plt.show()

if __name__ == '__main__':
	main()