import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('AAPL.csv')
X = data.iloc[:, 0].values.reshape(-1,1)
Y = data.iloc[:, 1].values.reshape(-1,1)

print("Analysing past 127 trading days of AAPL. Please wait.")

lr = LinearRegression()
lr.fit(X,Y)

Y_pred = lr.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()

userInput = input("How many days from now would you like to predict?")
pred_days = (int(userInput) + 127)

print("AAPL Price prediction " + str(userInput) + " days from now:")
print(lr.predict(np.array([pred_days]).reshape(-1, 1)))
