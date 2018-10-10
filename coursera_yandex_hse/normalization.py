from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from numpy import array

with open("/home/nik/Downloads/perceptron_train.csv", "r") as f:
    train = f.read().splitlines()

scaler = StandardScaler()
trainX = []
trainY = []

for feature in train:
    trainY.append(feature.split(',')[0])
    trainX.append(feature.split(',')[1:])

trainX = array(trainX).astype('float')
trainX_scaled = scaler.fit_transform(array(trainX).astype('float'))
trainY = array(trainY).astype('float')

with open("/home/nik/Downloads/perceptron_test.csv", "r") as f:
    test = f.read().splitlines()

testX = []
testY = []

for feature in test:
    testY.append(feature.split(',')[0])
    testX.append(feature.split(',')[1:])

testX = array(testX).astype('float')
testX_scaled = scaler.transform(array(testX).astype('float'))
testY = array(testY).astype('float')

clf = Perceptron(random_state=241)
clf.fit(trainX, trainY)

preds = clf.predict(testX)
a1 = accuracy_score(testY, preds)

clf.fit(trainX_scaled, trainY)

preds = clf.predict(testX_scaled)
a2 = accuracy_score(testY, preds)

a = round(a2-a1, 2)
with open("file", "w") as f:
    f.write("{}".format(a))


