import csv
import io
from flask import Flask, request, render_template, jsonify

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

        csv_reader = csv.reader(file_content)
        first_line = next(csv_reader)  # Get the first line of the CSV
        
        if len(first_line) < 2:
            return 'CSV file does not contain enough data', 400
        
        # Extract x and y coordinates from the first two items in the first line
        x = first_line[0]
        y = first_line[1]

        # Return the x and y coordinates as a JSON response
        return jsonify({'x': x, 'y': y})
    except Exception as e:
        return f"Error reading CSV: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
