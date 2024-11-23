import csv
import io
from flask import Flask, request, render_template, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML form

@app.route('/process_csv', methods=['POST'])
def process_csv():
    # Check if a file is part of the request
    if 'csv-file' not in request.files:
        return 'No file part', 400
    
    file = request.files['csv-file']

    if file.filename == '':
        return 'No selected file', 400

    # Read the file as CSV
    try:
        # Convert the binary stream to a text stream using io.TextIOWrapper
        # This makes it compatible with csv.reader
        file_content = io.StringIO(file.stream.read().decode('utf-8'))

        csv_df = pd.read_csv(file_content)
        print(csv_df.iloc[0]['p0_pos_y'])
        print(csv_df.iloc[0]['p5_pos_y'])

        # Extract x and y coordinates from the first two items in the first line
        coords = []
        for player in range(6):
            temp = []
            temp.append(csv_df.iloc[0]['p%i_pos_y' % player] * 5 + 512.0 + 88.0)
            temp.append(csv_df.iloc[0]['p%i_pos_x' % player] * 5 + 409.6)
            coords.append(temp)

        print(coords)

        temp = []
        temp.append(csv_df.iloc[0]['ball_pos_y'] * 5 + 512.0)
        temp.append(csv_df.iloc[0]['ball_pos_x'] * 5 + 409.6)
        coords.append(temp)

        # Return the x and y coordinates as a JSON response
        return jsonify(coords)
    except Exception as e:
        return f"Error reading CSV: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
