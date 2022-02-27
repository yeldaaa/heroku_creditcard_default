import flask
from flask import Flask, request, render_template
from graphviz import render
import numpy as np
import joblib
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template('index.html')

@app.route("/logreg", methods=["GET", "POST"])
def logreg():
    if request.method == "POST":
        income = request.form.get("income")
        age = request.form.get("age")
        loan = request.form.get("loan")
        model = joblib.load("Logistic Regression")
        logreg_arr = np.array([[float(income), float(age), float(loan)]])
        pred = model.predict(logreg_arr)
        if pred[0] == 0:
            s = ("It is predicted that this customer will not default")
        else:
            s = ("It is predicted that this customer will default")
        return(render_template("logreg.html", prediction= s))
    else:
        return render_template('logreg.html')


@app.route("/dectree", methods=["GET", "POST"])
def dectree():
    if request.method == "POST":
        income = request.form.get("income")
        age = request.form.get("age")
        loan = request.form.get("loan")
        model = joblib.load("Pruned Decision Tree")
        dectree_arr = np.array([[float(income), float(age), float(loan)]])
        pred = model.predict(dectree_arr)
        if pred[0] == 0:
            s = ("It is predicted that this customer will not default")
        else:
            s = ("It is predicted that this customer will default")
        return(render_template("dectree.html", prediction= s))
    else:
        return render_template('dectree.html')

@app.route("/randomforest", methods=["GET", "POST"])
def randomforest():
    if request.method == "POST":
        income = request.form.get("income")
        age = request.form.get("age")
        loan = request.form.get("loan")
        model = joblib.load("Random Forest (Gridsearch)")
        randomforest_arr = np.array([[float(income), float(age), float(loan)]])
        pred = model.predict(randomforest_arr)
        if pred[0] == 0:
            s = ("It is predicted that this customer will not default")
        else:
            s = ("It is predicted that this customer will default")
        return(render_template("randomforest.html", taste= s))
    else:
        return render_template('randomforest.html')

@app.route("/xgboost", methods=["GET", "POST"])
def xgboost():
    if request.method == "POST":
        income = request.form.get("income")
        age = request.form.get("age")
        loan = request.form.get("loan")
        model = joblib.load("XGBoost Grid Search")
        xgboost_arr = np.array([[float(income), float(age), float(loan)]])
        pred = model.predict(xgboost_arr)
        if pred[0] == 0:
            s = ("It is predicted that this customer will not default")
        else:
            s = ("It is predicted that this customer will default")
        return(render_template("xgboost.html", prediction= s))
    else:
        return render_template('xgboost.html')

@app.route("/nn", methods=["GET", "POST"])
def neuralnetwork():
    if request.method == "POST":
        income = request.form.get("income")
        age = request.form.get("age")
        loan = request.form.get("loan")
        model = joblib.load("Neural Network")
        nn_arr = np.array([[float(income), float(age), float(loan)]])
        pred = model.predict(nn_arr)
        if pred[0] == 0:
            s = ("It is predicted that this customer will not default")
        else:
            s = ("It is predicted that this customer will default")
        return(render_template("nn.html", prediction= s))
    else:
        return render_template('nn.html')



if __name__ == "__main__":
    app.run(debug=True)