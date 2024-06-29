from flask import Flask, request, render_template
import pandas as pd
from models.clustering import load_data, preprocess_data, apply_kmeans

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        data = load_data(file)
        data = preprocess_data(data)
        clustered_data, model = apply_kmeans(data)
        clustered_data.to_csv('static/customer_data.csv', index=False)
        return render_template('index.html', tables=[clustered_data.to_html(classes='data')], titles=clustered_data.columns.values)

if __name__ == '__main__':
    app.run(debug=True)
