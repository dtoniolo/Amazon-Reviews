import gzip
import json
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, \
                            classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import utils
from sklearn.feature_extraction.text import CountVectorizer

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
# extracting the texts and the scores
reviewsText = [review['reviewText'] for review in reviews]
scores = np.array([review['overall'] for review in reviews])
classes, class_counts = utils.class_counts(scores)
utils._print_class_counts(classes, class_counts)

# text preprocessing
vectorizer = CountVectorizer(binary=True, dtype=bool)
bow = vectorizer.fit_transform(reviewsText)

# train test split
split_data = train_test_split(bow, scores, test_size=1/3, shuffle=True,
                              stratify=scores)
train_x, test_x, train_y, test_y = split_data
sample_weights = utils.get_scores(class_counts, classes, train_y)

# training Naïve Bayes
print('Training the classifier... ', end=' .')
classifier = BernoulliNB()
classifier.fit(train_x, train_y, sample_weights)
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
