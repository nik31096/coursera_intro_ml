from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
from sys import exit

grid = {'C': np.power(10.0, np.arange(-5, 6))}
print("[INFO] data loading")
newsgroups = fetch_20newsgroups(subset='all', 
                          categories=["alt.atheism", "sci.space"])

vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(newsgroups.data)
#labels = vectorizer.transform(newsgroups.target)
labels = newsgroups.target
features_names = np.array(vectorizer.get_feature_names())
vocab = vectorizer.vocabulary_
print("[INFO] svm works")
#svm = SVC(kernel='linear', random_state=241)
#kf = KFold(n_splits=5, random_state=241)
#gs = GridSearchCV(svm, grid, scoring='accuracy', cv=kf, verbose=1)
#scores = cross_val_score(svm, features, labels, cv=kf, scoring='accuracy')
#cv_scores.append(scores.mean())
#gs.fit(features, labels)
#print("[INFO] fit on svm with best parameters C")
svm = SVC(kernel='linear', random_state=241, C=10)
svm.fit(features, labels)
cx = sp.sparse.coo_matrix(svm.coef_)
coef = {}
for _, integer, value  in zip(cx.row, cx.col, cx.data):
    coef[integer] = abs(value)

indices = [x[0] for x in sorted(coef.items(), key=lambda x: x[1])[-10:]]
words = []
for word, integer in vectorizer.vocabulary_.items():
    if integer in indices:
        words.append(word)
print(sorted(words))
with open('file', 'w') as f:
    f.write(','.join(sorted(words)))

