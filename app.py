from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), 'banglore_home_price_prediction_model.pkl')
model = pickle.load(open(model_path, 'rb'))



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])
    total_sqft = int(request.form['total_sqft'])

    # Assuming the model expects input in the following order: [location, total_sqft, bath, bhk]
    # You might need to preprocess the location if your model requires it (e.g., one-hot encoding)
    input_features = pd.DataFrame([[location, total_sqft, bath, bhk]], 
                                  columns=['location', 'total_sqft', 'bath', 'bhk'])

    # Predict the price
    predicted_price = model.predict(input_features)[0]

    return render_template('index.html', price=f"The predicted price of the house is {predicted_price:.2f} lakhs.")

if __name__ == '__main__':
    app.run(debug=True)
