import multiprocessing
import os
import time

import requests
import webview
import pandas as pd
import openpyxl
import shutil
from waitress import serve
from flask import Flask, render_template, request
from psychDiagnosis.psychopathy_main import generate_plots

app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')

# Function to load data from the Excel file
def load_data_from_excel(file_path):
    spreadsheet = pd.ExcelFile(file_path)
    scores_data = spreadsheet.parse('Scores')
    words_data = spreadsheet.parse('Words')

    criteria = scores_data.rename(columns={
        'Factors': 'name',
        'Weights': 'default_importance',
        'Scoring': 'description'
    }).to_dict(orient='records')

    scoring_criteria = {}
    for row in words_data.itertuples(index=False, name=None):
        if pd.notnull(row[0]):  # Ensure factor name is not NaN
            scoring_criteria[row[0]] = [value for value in row[1:] if pd.notnull(value)]

    return criteria, scoring_criteria


# Save updated data to Excel
def save_updates_to_excel(file_path, updated_criteria):
    workbook = openpyxl.load_workbook(file_path)
    scores_sheet = workbook['Scores']

    for row in scores_sheet.iter_rows(min_row=2, max_row=scores_sheet.max_row):
        factor = row[0].value  # Assuming the first column is "Factors"
        for item in updated_criteria:
            if item['name'] == factor:
                row[1].value = item.get('selected_importance', row[1].value)  # Update weights
                row[2].value = item.get('selected_score', row[2].value)  # Update scoring

    workbook.save(file_path)


PLOTS_DIR = os.path.join(app.root_path, 'frontend', 'static', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    EXCEL_FILE = os.path.join(os.path.dirname(__file__), 'psychDiagnosis', 'excel', 'PCLRWords.xlsx')
    EXCEL_FILE = os.path.abspath(EXCEL_FILE)
    criteria, scoring_criteria = load_data_from_excel(EXCEL_FILE)

    if request.method == 'POST':
        results = request.form.to_dict()
        updated_scores = {key: value for key, value in results.items() if key.endswith('_score')}
        updated_importance = {key: value for key, value in results.items() if key.endswith('_importance')}

        updated_criteria = []
        for item in criteria:
            name = item['name']
            updated_item = item.copy()
            updated_item['selected_score'] = updated_scores.get(f"{name}_score", "N/A")
            updated_item['selected_importance'] = updated_importance.get(f"{name}_importance", "N/A")
            updated_criteria.append(updated_item)

        save_updates_to_excel(EXCEL_FILE, updated_criteria)

        if os.path.exists(PLOTS_DIR):
            shutil.rmtree(PLOTS_DIR)
        os.makedirs(PLOTS_DIR, exist_ok=True)

        plot_paths = generate_plots(PLOTS_DIR, EXCEL_FILE)
        plot_urls = [os.path.join('static', 'plots', os.path.basename(path)) for path in plot_paths]

        return render_template('results.html', results=updated_criteria, plots=plot_urls)

    return render_template('index.html', criteria=criteria, scoring_criteria=scoring_criteria)


def run_server():
    """Run the Flask app with Waitress."""
    serve(app, host='127.0.0.1', port=5000)


def wait_for_server(host, port, timeout=10):
    """Wait for the server to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{host}:{port}")
            if response.status_code == 200:
                return True
        except requests.ConnectionError:
            time.sleep(0.1)  # Retry every 100ms
    raise TimeoutError(f"Server did not start within {timeout} seconds.")


if __name__ == '__main__':
    server_process = multiprocessing.Process(target=run_server)
    server_process.start()

    try:
        # Wait for the server to be ready
        wait_for_server("127.0.0.1", 5000)

        # Launch the webview window
        webview.create_window("Psychopathy Diagnosis", "http://127.0.0.1:5000", width=1500, height=800)
        webview.start()
    except TimeoutError as e:
        print(e)
    finally:
        print("Shutting down server...")
        server_process.terminate()
        server_process.join()