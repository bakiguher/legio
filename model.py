import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression
import joblib

data=pd.read_csv('legionall.csv')

X = data[['soldier','mule','slave']].to_numpy()
y = np.array(data[['water','grain','fodder']]).reshape(-3, 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=92)
classifier = LinearRegression() 
classifier.fit(X_train, y_train)
pred=classifier.predict(X_test)

mse = metrics.mean_squared_error(y_test, pred)
print(classifier.score(X_test, y_test))

joblib.dump(classifier, "clf.pkl")