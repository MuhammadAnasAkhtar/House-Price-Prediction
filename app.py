from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model
model = joblib.load('house_price_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sqft_living = float(request.form['sqft_living'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        floors = float(request.form['floors'])
        sqft_lot = float(request.form['sqft_lot'])
        sqft_above = float(request.form['sqft_above'])
        sqft_basement = float(request.form['sqft_basement'])

        # Create a numpy array for the input features
        input_data = np.array([[sqft_living, bedrooms, bathrooms, floors, sqft_lot, sqft_above, sqft_basement]])
        
        # Get the prediction
        prediction = model.predict(input_data)

        return render_template('index.html', prediction_text=f'Predicted House Price: ${round(prediction[0], 2)}')

if __name__ == '__main__':
    app.run(debug=True)
