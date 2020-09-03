import gzip
import string
import json
from tqdm import tqdm
import numpy as np
import nltk


# read file
gzipped = False
filename = '../Clothing_Shoes_and_Jewelry_5.json'
if gzipped:
    file_reader = gzip.open(filename, 'r')
    reviews = [json.loads(review.decode('utf-8')) for review in file_reader]
else:
    with open(filename, 'r') as f:
        reviews = [json.loads(review) for review in f.readlines()]

reviews = reviews[:100]

# word tokenizer
print('Tokenizing:')
for review in tqdm(reviews):
    review['reviewText'] = nltk.tokenize.word_tokenize(review['reviewText'])

# stop words removal
print('Removing stop words:')
stop_words = set(nltk.corpus.stopwords.words('english'))
for review in reviews:
    review['reviewText1'] = list()
    for token in review['reviewText']:
        if token not in stop_words:
            review['reviewText1'].append(token)

# stemming
print('Stemming:')
ps = nltk.stem.PorterStemmer()
for review in tqdm(reviews):
    review['reviewText2'] = list()
    for token in review['reviewText1']:
        review['reviewText2'].append(ps.stem(token))

# remuving puntuation
print('Removing punctuation:')
for review in tqdm(reviews):
    review['reviewText3'] = list()
    for token in review['reviewText2']:
        if token not in string.punctuation:
            review['reviewText3'].append(token)

# building the vocabulary
print('Building the vocabulary:')
vocabulary = set()
for review in tqdm(reviews):
    for token in review['reviewText3']:
        vocabulary.add(token)
vocabulary = list(vocabulary)
vocabulary = np.array(vocabulary)

# building the Bag of Words representation
print('Building the Bag of Words representation:')
bow = np.zeros(shape=(len(vocabulary), len(reviews)),
               dtype='bool')
for j, review in enumerate(tqdm(reviews)):
    for i, token in enumerate(review['reviewText3']):
        i = np.where(vocabulary == token),
        i = i[0]
        bow[i, j] = True

# training Naïve Bayes
print('Training the classifier... ')
train_set = [(dict(zip(vocabulary, feat)), str(review['overall']))
             for (feat, review) in zip(bow.T, reviews)]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print('Accuracy', nltk.classify.accuracy(classifier, train_set))
