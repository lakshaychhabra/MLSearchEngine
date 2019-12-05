from flask import Flask, render_template, url_for, request, redirect
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import sqlite3
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import re
import os
from sqlalchemy import create_engine # database connection
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
import scipy.sparse
import pickle

# Model
filename_model = 'clf_final.sav'
clf_final = joblib.load(filename_model)

filename_tfidf = 'tfidf.sav'
tfidf = joblib.load(filename_tfidf)

# Y_i
filename_y = 'y'
infile = open(filename_y,'rb')
y = pickle.load(infile)
# Tfidf Data
data = scipy.sparse.load_npz('data.npz')


con = sqlite3.connect('dataset/processed.db')
processed = pd.read_sql_query("""SELECT * FROM processed""", con)
con.close()

# labels = {"c#" : 0, "java" : 1, "c++" : 2, "c" : 3, "ios" : 4}
labels_map = { 0 : "c#" , 1 : "java" , 2 : "c++" , 3 : "c", 4 : "ios"}

def process_query(query):
    preprocessed_reviews = []
    sentance = re.sub("\S*\d\S*", "", query).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords.words('english'))
    preprocessed_reviews.append(sentance.strip())
    return preprocessed_reviews

def tfidf_search(query):
    query = process_query(query)
    query_trans = tfidf.transform(query)
    pairwise_dist = pairwise_distances(data, query_trans)
    
    indices = np.argsort(pairwise_dist.flatten())[0:10]
    df_indices = list(processed.index[indices])
    return df_indices


def label(query):
    query = process_query(query)
    query = tfidf.transform(query)
    ans = clf_final.predict(query)
    return labels_map[ans[0]]


def change_query(query):
    tag = label(query)
    return query + " " + tag


def enter_queries(query) : 
    vals = []
    print("The Query is :", query)
    query = change_query(query)
    df_indices = tfidf_search(query)
    print("The Model Interpreted Query is :", query)
    print("Top Results : ")
    for i in (df_indices):
        # print("Title : ", processed.Title.iloc[i])
        vals.append(processed["Title"].iloc[i])
    return vals, query


# query = "synchronization"

app = Flask(__name__)

@app.route('/', methods = ['POST', 'GET'])


def index():
    if request.method == 'POST':
        task_content = request.form['content']
        print(task_content)
        i,j = enter_queries(task_content)
        f =  i
        k = j
        # task_content = "heyyyyyy"
        return render_template("index.html", tasks = [f,k])
    else:
        task_content = ""
        # task = task_content
        return render_template("index.html", tasks = [" ", " "])

if __name__ == "__main__":
    app.run(debug=True,  port=8080)




