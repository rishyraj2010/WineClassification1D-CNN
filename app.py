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
    '''
    For rendering results on HTML GUI
    '''
    flt_features = [float(x) for x in request.form.values()]
    final_features = [np.array(flt_features)]
    model = tf.keras.models.load_model('white_wine_pipeline_1DCNN.keras')
    output = np.argmax(model.predict(([final_features[x]] for x in final_features), axis=1)

    return render_template('index.html', prediction_text='Wine Quality is $ {}'.format(output))


if __name__ == "__main__":
    app.run(port=5000, debug=True)
