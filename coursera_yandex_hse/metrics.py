from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold, cross_val_score
from numpy import linspace, array
from matplotlib import pyplot as plt

data, target = load_boston(return_X_y=True)
data = scale(data).astype('float')
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for p in linspace(1, 10, 200):
    knn = KNeighborsRegressor(n_neighbors=5, weights="distance", p=p)
    scores = cross_val_score(knn, data, target, cv=kf, scoring="neg_mean_squared_error")
    cv_scores.append(scores.mean())

print(linspace(1, 10, 200)[cv_scores.index(max(cv_scores))], max(cv_scores))
with open("file", "w") as f:
    f.write("{}".format(linspace(1, 10, 200)[cv_scores.index(max(cv_scores))]))

plt.plot(linspace(1, 10, 200), cv_scores)
plt.show()
