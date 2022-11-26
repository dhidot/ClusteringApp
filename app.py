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
    distances = []  # list untuk menampung jarak antara nilai input dengan nilai yang ada di dataset

    for i, this_segment in enumerate(model):
        dist = distance.cityblock(vals, this_segment)
        distances.append(dist)
        indexMin = np.argmin(distances)
        assignedCluster.append(indexMin)

    if (indexMin == 0):
        result = 'cluster 1'
    elif (indexMin == 1):
        result = 'cluster 2'
    else:
        result = 'cluster 3'

    return render_template('predict.html', resultValue=f'Data yang diinput termasuk kedalam {result}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
