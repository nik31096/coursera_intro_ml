from sklearn.svm import SVC

with open("/home/nik/Downloads/svm_data.csv", "r") as f:
    data = f.read().splitlines()

features = []
labels = []

for feature in data:
    labels.append(feature.split(',')[0])
    features.append(feature.split(',')[1:])

svm = SVC(C=100000, kernel='linear')
svm.fit(features, labels)

print(svm.support_)
print(svm.support_vectors_)

with open('file', "w") as f:
    f.write(",".join([str(x+1) for x in svm.support_]))

