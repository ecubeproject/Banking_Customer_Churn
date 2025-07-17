# Use virtual environment Python 3.9 (Customer_Churn2)
import subprocess
import os
import pandas as pd
import joblib
import io
from flask import Flask, render_template, redirect, request, jsonify,url_for
from sklearn.model_selection import train_test_split
from werkzeug.utils import secure_filename
from shap_analysis import load_model,shap_analysis_function
from preprocessing import CategoricalEncoder, AddFeatures

# Initialize Flask application
app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/shap_analysis', methods=['GET', 'POST'])
def shap_analysis():
    file_data = None
    message = None  # To hold error or success messages

    if request.method == 'POST':
        file = request.files.get('file', None)
        if file and file.filename != '':
            filename = secure_filename(file.filename)  # Use Flask's secure_filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)  # Save the file to the UPLOAD_FOLDER

            # Read from the saved file for processing and display
            df = pd.read_csv(filepath)
            file_data = df.to_html(classes='data', header="true")
            df.to_csv('X_train.csv', index=False)  # If needed, save again under a different name
        else:
            message = 'First select a file using Browse button then press Upload'
    elif request.method == 'GET':
        # Clear any previous session data or file uploads if reset is triggered
        file_data = None
    return render_template('shap.html', file_data=file_data, message=message)


@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        print(file.filename)
        filename = 'Churn_Modelling.csv'  # You can use a dynamic name if needed
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'message': 'File uploaded successfully', 'filepath': filepath})

    return jsonify({'error': 'File upload failed'})


@app.route('/perform_shap_analysis', methods=['POST'])
def perform_shap_analysis():
    try:
        # Attempt to get JSON data from the request
        data = request.get_json()
        if not data or 'customer_id' not in data:
            return jsonify({'error': 'No data uploaded or CustomerID not present in data'}), 400

        customer_id = int(data['customer_id'])

        # Load the data
        df = pd.read_csv('uploaded_files/Churn_Modelling.csv')
        y = df['Exited'].values
        X= df.drop(['Exited'], axis=1)

        # Find the index of the row that matches the CustomerID
        matching_indices = df.index[df['CustomerId'] == customer_id].tolist()
        if not matching_indices:
            return jsonify({'error': 'Customer ID not found'}), 404
        row_index = matching_indices[0]  # Get the first (and should be only) matching index

        # Assuming model and other setups are handled previously or elsewhere
        model = load_model()  # Load your model (ensure this function is defined or replace with actual loading code)

        # Generate SHAP plot
        plot_url = shap_analysis_function(row_index, customer_id,X,y, model)
        print(f"Generated SHAP plot at URL: {plot_url}")

        return jsonify({'shapImageUrl': plot_url})

    except FileNotFoundError:
        print("File not found at 'uploaded_files/Churn_Modelling.csv'")
        return jsonify({'error': 'File not found'}), 404
    except ValueError as ve:
        print(f"Value error: {ve}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"Unhandled exception: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/analyse_churn', methods=['POST'])
def analyse_churn_route():
    # Running the Python script as a separate process
    subprocess.run(['python', 'analyse_churn.py'])
    return jsonify({"message": "Analysis started, please check the new tab for results."})


@app.route('/results_page')
def results_page():
    stats_html = ""
    images = []

    # Check if the file exists before trying to read it
    stats_file_path = 'static/Age_stats.csv'
    if os.path.exists(stats_file_path):
        stats_data = pd.read_csv(stats_file_path)
        stats_html = stats_data.to_html()

    # Load images if any are present
    if os.path.exists('static'):
        image_files = os.listdir('static')
        images = [file for file in image_files if file.endswith('.png')]

    return render_template('results_page.html', stats_html=stats_html, images=images)

# Load the pre-trained model
model = joblib.load('final_churn_model_f1_0_53.sav')

# Preprocessing and feature engineering functions
def preprocess_input(data):
    data_preprocessed = model.named_steps['categorical_encoding'].transform(data)
    data_preprocessed = model.named_steps['add_new_features'].transform(data_preprocessed)
    return data_preprocessed

@app.route('/', methods=['GET', 'POST'])
def index():
    file_data = None
    high_churn_list = None
    message = None  # To hold error or success messages

    if request.method == 'POST':
        file = request.files.get('file', None)
        if file and file.filename != '':
            X_test = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
            file_data = X_test.to_html(classes='data', header="true")
            X_test.to_csv('X_test.csv', index=False)
        else:
            message = 'First select a file using Browse button then press Upload'
    elif request.method == 'GET':
        # Clear any previous session data or file uploads if reset is triggered
        file_data = None
        high_churn_list = None

    return render_template('index.html', file_data=file_data, high_churn_list=high_churn_list, message=message)


@app.route('/high_churn_list', methods=['POST'])
def high_churn_list():
    X_test = pd.read_csv('X_test.csv')
    X_test_preprocessed = preprocess_input(X_test)
    file_data = X_test.to_html(classes='data', header="true")

    try:
        business_decided_probability = float(request.form['probability'])
    except ValueError:
        return redirect('/')

    pred_probs = model.named_steps['classifier'].predict_proba(X_test_preprocessed)[:, 1]
    high_churn = X_test[pred_probs > business_decided_probability].copy()
    high_churn['prediction_probability'] = pred_probs[pred_probs > business_decided_probability]

    # Save the high churn list to CSV
    high_churn_sorted = high_churn.sort_values(by='prediction_probability', ascending=False)
    high_churn_sorted.to_csv('high_churn_list.csv', index=False)  # Make sure to save without index

    high_churn_list_html = high_churn_sorted.to_html(classes='data', header="true")

    return render_template('index.html', file_data=file_data, high_churn_list=high_churn_list_html)


if __name__ == '__main__':
        app.run(debug=True, port=5000)
