from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np


#initializing the flask app
app = Flask(__name__)


#calling the models
reg_model = open('flask_model.pkl',"rb")
ml_model = joblib.load(reg_model)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    print("The Selling Price of Car is")
    if request.method == 'POST':
        print(request.form.get('Kilometer'))
        try:
            Kilometer = float(request.form['Kilometer'])
            Fuel = float(request.form['Fuel'])
            Seller = float(request.form['Seller'])
            Transmission = float(request.form['Transmission'])
            Onwer = float(request.form['Onwer'])
            Mileage = float(request.form['Mileage'])
            Engine = float(request.form['Engine'])
            Max = float(request.form['Max'])
            Seats = float(request.form['Seats'])
            predictors = [Kilometer, Fuel, Seller, Transmission, Onwer, Mileage, Engine, Max, Seats]
            pred_args_arr = np.array(predictors)
            pred_args_arr = pred_args_arr.reshape(1, -1)

            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction), 2)

        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('result.html', prediction = model_prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0')


