import joblib
model = joblib.load("Logistic Regression")

pred = model.predict([[float(10000),float(55),float(12312398)]])
print(pred[0])