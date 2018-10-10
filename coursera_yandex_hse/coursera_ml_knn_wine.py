from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt
from numpy import array, average

with open("wine.data", "r") as f:
    data = f.read().splitlines()

labels = []
features = []

for record in data:
    labels.append(record.split(',')[0])
    features.append(record.split(',')[1:])

features = array(features).astype('float')
features_scaled = array(scale(features)).astype('float')
labels = array(labels).astype('float')

#trainX, testX, trainY, testY = train_test_split(features, labels, test_size=0.25)
cv_scores = []
for k in range(1, 51):
    knn = KNeighborsClassifier(n_neighbors=k)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(knn, features, labels, cv=kf, scoring='accuracy')
    cv_scores.append(scores.mean())
    
plt.plot(list(range(1, 51)), cv_scores)
a = max(cv_scores)
b = cv_scores.index(max(cv_scores)) + 1
print(a, b)

with open('file', "w") as f:
    #f.write("{}".format(round(a, 2)))
    f.write("{}".format(b))

plt.show()
