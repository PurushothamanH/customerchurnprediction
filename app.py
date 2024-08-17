from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for home
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    else:
        try:
            # Collect data from form
            data = CustomData(
                gender=request.form.get('gender'),
                SeniorCitizen=int(request.form.get('SeniorCitizen', 0)),  # Convert to integer
                Partner=request.form.get('Partner'),
                Dependents=request.form.get('Dependents'),
                tenure=int(request.form.get('tenure', 0)),  # Convert to integer
                PhoneService=request.form.get('PhoneService'),
                MultipleLines=request.form.get('MultipleLines'),
                InternetService=request.form.get('InternetService'),
                OnlineSecurity=request.form.get('OnlineSecurity'),
                OnlineBackup=request.form.get('OnlineBackup'),
                DeviceProtection=request.form.get('DeviceProtection'),
                TechSupport=request.form.get('TechSupport'),
                StreamingTV=request.form.get('StreamingTV'),
                StreamingMovies=request.form.get('StreamingMovies'),
                Contract=request.form.get('Contract'),
                PaperlessBilling=request.form.get('PaperlessBilling'),
                PaymentMethod=request.form.get('PaymentMethod'),
                MonthlyCharges=float(request.form.get('MonthlyCharges', 0.0)),  # Convert to float
                TotalCharges=float(request.form.get('TotalCharges', 0.0))  # Convert to float
            )

            # Convert data to DataFrame
            pred_df = data.get_data_as_data_frame()
            # print(pred_df)

            # Load prediction pipeline and get results
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            # Render results in home.html
            return render_template('home.html', results=results[0])

        except Exception as e:
            print(f"Error occurred: {e}")
            return render_template('home.html', results="Error during prediction. Please check input values.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
