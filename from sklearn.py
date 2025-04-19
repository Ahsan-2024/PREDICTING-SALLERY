from sklearn.linear_model import LinearRegression
import numpy as np

# Experience in years
X = np.array([[1], [2], [3], [4], [5]])
# Salary in $1000
y = np.array([30, 35, 45, 50, 60])

model = LinearRegression()
model.fit(X, y)

# Predict salary for 6 years of experience
print("Predicted salary for 6 years:", model.predict([[6]])[0])
