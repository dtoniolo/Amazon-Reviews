import gzip
import json
from nltk.sentiment import SentimentIntensityAnalyzer
from math import sqrt
from tqdm import tqdm
import plotly.express as px


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
    review['score'] = transform_score(review['sentimentScore']['compound'])
    error += (review['score'] - review['overall'])**2
error /= len(reviews)
error = sqrt(error)

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
