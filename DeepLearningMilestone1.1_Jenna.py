# -*- coding: cp1252 -*-
import random
import csv
import json
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
#from eli5 import show_weights

training_data = '/home/jjpelt/fincore-train.tsv.1' #nämä täytyy muuttaa oman kirjaston mukaiseksi
testing_data = '/home/jjpelt/fincore-test.tsv.1' #-"-
dev_data = '/home/jjpelt/fincore-dev.tsv.1' #-"-



def class_counts(df, label='class'):
    return df[label].value_counts().to_string(header=None)

#separate class, id, etc information from each line
train_data = pd.read_csv(training_data, sep='\t', names=('class', 'text'))
train_data = train_data[['class', 'text']]   
test_data = pd.read_csv(testing_data, sep='\t', names=('class', 'text'))
test_data = test_data[['class', 'text']] 
devel_data = pd.read_csv(dev_data, sep='\t', names=('class', 'text'))
devel_data = devel_data[['class', 'text']] 

print("Feature counts:")
print(class_counts(train_data))

train_Y, train_texts = train_data['class'], train_data['text']
devel_Y, devel_texts = devel_data['class'], devel_data['text']
test_Y, test_texts = test_data['class'], test_data['text']

print(train_data[0:10])

space_tokenizer = lambda text: text.split()

vectorizer = TfidfVectorizer(tokenizer=space_tokenizer, ngram_range=(1,2))
vectorizer.fit(train_texts)

train_X = vectorizer.transform(train_texts)
devel_X = vectorizer.transform(devel_texts)
test_X = vectorizer.transform(test_texts)

#Distribution of texts and classes in the dataset

print("Train:", len(train_texts))
print(class_counts(train_data))

print("Devel:",len(devel_texts))
print(class_counts(devel_data))

print("Test:",len(test_texts))
print(class_counts(test_data))


from sklearn import metrics

#train_results = []
#devel_results = []
results = []

for c in (0.001, 0.01, 0.1, 1): #10, 100 out, need for speed
    classifier = LinearSVC(C=c, class_weight=None, max_iter=100000, loss="squared_hinge")
    classifier.fit(train_X, train_Y)
    pred_train = classifier.predict(train_X)
    pred_devel = classifier.predict(devel_X)
    print("C: ",c, "Train: ", metrics.f1_score(train_Y, pred_train, average='micro'), "Devel: ", metrics.f1_score(devel_Y, pred_devel, average='micro'))
    results.append({"C": c, "Train F": metrics.f1_score(train_Y, pred_train, average='micro'), "Devel F": metrics.f1_score(devel_Y, pred_devel, average='micro')})
