import numpy as np

from flask import Flask, request, jsonify, render_template

import tensorflow as tf

# Assuming you saved the model using model.save

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    import numpy as np
from flask import Flask, render_template, request
import tensorflow as tf

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Assuming the form fields contain features as input
    flt_features = [float(x) for x in request.form.values()]
    final_features = [np.array(flt_features)]
    
    # Load the model
    model = tf.keras.models.load_model('white_wine_pipeline_1DCNN.keras')

    # Predict using the loaded model
    output = np.argmax(model.predict(np.array(final_features)), axis=1)

    return render_template('index.html', prediction_text='Wine Quality is $ {}'.format(output[0]))

if __name__ == '__main__':
    app.run(debug=True)
