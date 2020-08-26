import gzip
import json
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression


def transform_score(score):
    return (score + 4)/2 + 1


# read file
gzipped = False
filename = 'Clothing_Shoes_and_Jewelry_5.json'
if gzipped:
    file_reader = gzip.open(filename, 'r')
    reviews = [json.loads(review.decode('utf-8')) for review in file_reader]
else:
    with open(filename, 'r') as f:
        reviews = [json.loads(review) for review in f.readlines()]

# compute model score and msqe
error = 0
sid = SentimentIntensityAnalyzer()
for review in tqdm(reviews):
    review['sentimentScore'] = sid.polarity_scores(review['reviewText'])

scores = [review['sentimentScore']['compound'] for review in reviews]
scores = np.array(scores).reshape((-1, 1))
scores_squared = scores**2
x = np.hstack((scores, scores_squared))
target_scores = [review['overall'] for review in reviews]
target_scores = np.array(target_scores)
lr = LinearRegression(n_jobs=-1)
lr.fit(x, target_scores)
lr.score(x, target_scores)


# plot model score vs user score
overalls = [str(review['overall']) for review in reviews]
scores = [review['score'] for review in reviews]
plot = px.violin(x=overalls, y=scores)
plot.update_layout(title={'text': "1st Experiment",
                          'x': 0.5,
                          'xanchor': 'center',
                          'yanchor': 'top'})
plot.update_xaxes(title='User Score')
plot.update_yaxes(title='Model Score', range=[0.5, 5.5])
