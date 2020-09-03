import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, \
                            classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import utils
import nlp
import tensorflow.keras as keras


filename = 'reviews_Clothing_Shoes_and_Jewelry_5.json.gz'
reviews = utils.read_file(filename)

reviews = reviews[:100]
scores = [review['overall'] for review in reviews]
scores = np.array(scores)
reviews = [review['reviewText'] for review in reviews]

reviews, scores = utils.transform_to_binary_classification(reviews, scores)

# NLP pipeline
reviews = nlp.tokenize(reviews)
reviews = nlp.remove_stopwords(reviews)
reviews = nlp.lemmatize(reviews)
reviews = nlp.remove_punctuation(reviews)


# building the Tf-idf representation
print('Building the Tf-idf representation...', end=' .')
vectorizer = TfidfVectorizer(lowercase=False)
tfidf = vectorizer.fit_transform(reviews)
print('Done')

# LSA
print('Decomposing the Tf-idf binary array... ', end=' .')
n_features = 50
decomposer = TruncatedSVD(n_features)
dc_tfidf = decomposer.fit_transform(tfidf)
del tfidf
print('Done')

# counting class frequencies
classes, class_counts = utils.class_counts(scores)
utils._print_class_counts(classes, class_counts)
weights = utils.get_class_scores(classes, class_counts)
class_weights = dict(zip(classes, weights))

# train test split
scores = keras.utils.to_categorical(scores)
split_data = train_test_split(dc_tfidf, scores, test_size=1/3, shuffle=True,
                              stratify=scores)
del dc_tfidf
train_x, test_x, train_y, test_y = split_data


# defining the NN
classifier = keras.models.Sequential()
classifier.add(keras.layers.Dense(512, activation='relu',
                                  input_shape=(n_features,)))
classifier.add(keras.layers.Dense(128, activation='relu'))
classifier.add(keras.layers.Dense(2, activation='softmax'))
classifier.compile(loss='binary_crossentropy')
classifier.fit(train_x, train_y, class_weight=class_weights, batch_size=32,
               epochs=5)

# training the NN

estimates = classifier.predict(test_x)
estimates = np.argmax(estimates, axis=1)
test_y = np.argmax(test_y, axis=1)
print('Building the confusion matrix...', end=' .')
cm = confusion_matrix(test_y, estimates, labels=[1.0, 2.0, 3.0, 4.0, 5.0])
print('Done')

fig, ax = plt.subplots()
plot_confusion_matrix(classifier, test_x, test_y, labels=[1.0, 2.0,
                      3.0, 4.0, 5.0], ax=ax)
ax.set_title('Confusion Matrix')
fig.show()

print(classification_report(test_y, estimates))
