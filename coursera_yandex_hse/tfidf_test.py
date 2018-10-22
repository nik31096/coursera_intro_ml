from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'Это первый документ',
    'Этот документ второй',
    'Это третий документ',
    'Это первый документ?',
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(vectorizer.vocabulary_)
print(X.shape)
