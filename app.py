from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load('model.joblib')

# Load the structure of the dataframe
df_structure = pd.read_pickle('df_structure.pkl')


def predict_analyst_category(model, features, target_names):
    """
    Predict the analyst category ('Buy', 'Hold', or 'Sell') based on;
    'Return_On_Equity(%)', 'OCI', 'Combined_ratio_P&C_and_Disabilty(%)', 'Outstanding_shares(weighted_average)', 
    'Open', 'Adj Close', 'Volume' using the trained model.

    :param model: Trained MultiOutputRegressor model.
    :param features: dataframe that holds all features used for training the model
    :param target_names: List of target variable names in the order they were used during model training.
    :param feature_names: List of feature names as they were used during model training.
    :return: Predicted category.
    """
    # Create a DataFrame for the input features with the correct column names
    input_df = features

    # Predict using the model
    predictions = model.predict(input_df)[0]

    # Map predictions to their corresponding target names
    prediction_dict = dict(zip(target_names, predictions))

    # Aggregate predictions
    buy_prediction = prediction_dict['Buy_%'] + prediction_dict['Outperform_%']
    sell_prediction = prediction_dict['Sell_%'] + prediction_dict['Underperform_%']
    hold_prediction = prediction_dict['Hold_%']

    categories = {'Buy': buy_prediction, 'Hold': hold_prediction, 'Sell': sell_prediction}
    predicted_category = max(categories, key=categories.get)

    return predicted_category



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form submission
        form_data = request.form.to_dict()
        form_data = {key: [float(value)] for key, value in form_data.items()}  # Convert all values to float
        input_data = pd.DataFrame(form_data)
        
        # Ensure the input data matches the structure of the dataframe used in training
        for col in df_structure.columns:
            if col not in input_data.columns:
                input_data[col] = [0]  # Add missing columns as zeros
        
        # Use the same target names as used in training
        target_names = ['Buy_%', 'Outperform_%', 'Hold_%', 'Underperform_%', 'Sell_%']
        
        # Make prediction
        predicted_category = predict_analyst_category(model, input_data, target_names)
        
        # Return the result to the user
        return render_template('index.html', prediction=f'Predicted Analyst Consensus: {predicted_category}')
    except Exception as e:
        return render_template('index.html', prediction=f'Error: {str(e)}')



if __name__ == '__main__':
    app.run(debug=True)