import gzip
import string
import json
from tqdm import tqdm
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, \
                            classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import print_class_counts

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
    review['reviewText3'] = ' '.join(review['reviewText3'])


# building the Bag of Words representation
print('Building the Bag of Words representation:')
reviewsText = [review['reviewText3'] for review in reviews]
vectorizer = CountVectorizer(binary=True, dtype=bool)
bow = vectorizer.fit_transform(reviewsText)
scores = [review['overall'] for review in reviews]
print_class_counts(scores)

# train test split
split_data = train_test_split(bow, scores, test_size=1/3, shuffle=True,
                              stratify=scores)
train_x, test_x, train_y, test_y = split_data

# training Naïve Bayes
print('Training the classifier... ', end=' .')
classifier = BernoulliNB()
classifier.fit(train_x, train_y)
print('Done')
estimates = classifier.predict(test_x)
print('Building the confusion matrix...', end=' .')
cm = confusion_matrix(test_y, estimates, labels=[1.0, 2.0, 3.0, 4.0, 5.0])
print('Done')

fig, ax = plt.subplots()
plot_confusion_matrix(classifier, test_x, test_y, labels=[1.0, 2.0, 3.0, 4.0,
                      5.0], ax=ax)
ax.set_title('Confusion Matrix')
fig.show()

print(classification_report(test_y, estimates))
