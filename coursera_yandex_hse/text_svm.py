from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from matplotlib import pyplot as plt

c_range = [pow(10, x) for x in range(-5, 6)]
print("[INFO] data loading")
newsgroups = fetch_20newsgroups(subset='all', 
                          categories=["alt.atheism", "sci.space"])

vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(newsgroups.data)
labels = newsgroups.target
print(labels)
cv_scores = []
print("[INFO] svm works")
for C in c_range:
    svm = SVC(C=C, kernel='linear', random_state=241)
    kf = KFold(n_splits=5, random_state=241)
    scores = cross_val_score(svm, features, labels, cv=kf, scoring='accuracy')
    cv_scores.append(scores.mean())

print("[INFO] optimC found, fitting on svm")
optimC = c_range[cv_scores.index(max(cv_scores))]
print(optimC)
svm = SVC(C=optimC, kernel='linear', random_state=241)
svm.fit(features, labels)
best_10 = []
for (x, y), z in svm.coef_[:10]:
    best_10.append(newsgroups.data[y])
print(best_10)

print(c_range[cv_scores.index(max(cv_scores))], max(cv_scores))
#plt.plot([pow(10, x) for x in range(-5, 6)], cv_scores)
#plt.show()

