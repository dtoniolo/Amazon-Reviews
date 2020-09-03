import gzip
import string
import json
from tqdm import tqdm
import numpy as np
import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, \
                            classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import utils


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, nltk.wordnet.NOUN)


# read file
gzipped = False
filename = '../Clothing_Shoes_and_Jewelry_5.json'
if gzipped:
    file_reader = gzip.open(filename, 'r')
    reviews = [json.loads(review.decode('utf-8')) for review in file_reader]
else:
    with open(filename, 'r') as f:
        reviews = [json.loads(review) for review in f.readlines()]

#reviews = reviews[:100]

# word tokenizer
print('Tokenizing:')
for review in tqdm(reviews):
    review['reviewText'] = nltk.tokenize.word_tokenize(review['reviewText'])

# stop words removal
print('Removing stop words:')
stop_words = set(nltk.corpus.stopwords.words('english'))
for review in tqdm(reviews):
    review['reviewText1'] = list()
    for token in review['reviewText']:
        if token not in stop_words:
            review['reviewText1'].append(token)

# Lemmatization
print('Lemmatizing:')
wnl = nltk.stem.WordNetLemmatizer()
for review in tqdm(reviews):
    review['reviewText2'] = list()
    for token in review['reviewText1']:
        review['reviewText2'].append(wnl.lemmatize(token,
                                                   get_wordnet_pos(token)))

# remuving puntuation
print('Removing punctuation:')
for review in tqdm(reviews):
    review['reviewText3'] = list()
    for token in review['reviewText2']:
        if token not in string.punctuation:
            review['reviewText3'].append(token)
    review['reviewText3'] = ' '.join(review['reviewText3'])


# building the Tf-idf representation
print('Building the Tf-idf representation...', end=' .')
reviewsText = [review['reviewText3'] for review in reviews]
scores = [review['overall'] for review in reviews]
scores = np.array(scores)
binary = True
if binary:
    neg = np.where(scores < 3.0)[0]
    pos = np.where(scores > 3.0)[0]
    neu = np.where(scores == 3.0)[0]
    scores[neg] = 0
    scores[pos] = 1
    reviewsText = reviewsText[np.logical_not(neu)]
    scores = scores[np.logical_and(neu)]
del reviews
vectorizer = TfidfVectorizer(lowercase=False)
tfidf = vectorizer.fit_transform(reviewsText)
print('Done')

# LSA
print('Decomposing the Tf-idf binary array... ', end=' .')
decomposer = TruncatedSVD(100)
dc_tfidf = decomposer.fit_transform(tfidf)
del tfidf
print('Done')

# counting class frequencies
classes, class_counts = utils.class_counts(scores)
utils._print_class_counts(classes, class_counts)

# train test split
split_data = train_test_split(dc_tfidf, scores, test_size=1/3, shuffle=True,
                              stratify=scores)
del dc_tfidf
train_x, test_x, train_y, test_y = split_data
sample_weights = utils.get_scores(class_counts, classes, train_y)

# training the Random Forest
print('Training the classifier... ', end=' .')
classifier = RandomForestClassifier()
classifier.fit(train_x, train_y, sample_weights)
print('Done')
estimates = classifier.predict(test_x)
print('Building the confusion matrix...', end=' .')
cm = confusion_matrix(test_y, estimates, labels=[1.0, 2.0, 3.0, 4.0, 5.0])
print('Done')

fig, ax = plt.subplots()
plot_confusion_matrix(classifier, test_x, test_y, labels=[1.0, 2.0,
                      3.0, 4.0, 5.0], ax=ax)
ax.set_title('Confusion Matrix')
fig.show()

print(classification_report(test_y, estimates))
