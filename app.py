from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    url_for,
)
import pickle
import numpy as np
from scipy.spatial import distance

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    inputValues = request.form

    if inputValues is not None:
        # Ambil nilai dari form dan masukkan ke dalam array untuk diproses nantinya
        vals = []
        for key, value in inputValues.items():
            vals.append(int(value))

    # Lakukan prediksi
    with open('centroids.pkl', 'rb') as f:
        model = pickle.load(f)

    assignedCluster = []
    distances = [] # list untuk menampung jarak antara nilai input dengan nilai yang ada di dataset

    for i, this_segment in enumerate(model):
        dist = distance.cityblock(vals, this_segment)
        distances.append(dist)
        indexMin = np.argmin(distances)
        assignedCluster.append(indexMin)

    return render_template('index.html', resultValue=f'Data yang diinput termasuk kedalam cluster nomor {indexMin, assignedCluster}')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
