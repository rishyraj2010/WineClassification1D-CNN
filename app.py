from flask import Flask,request,render_template,url_for
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib as joblib
import os

model=joblib.load('iris_model_LR.pkl')
scaler=joblib.load('scaler.save')

app =Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/',methods=['GET','POST'])
def home():
    if request.method =='POST':
        model = tf.keras.models.load_model('white_wine_pipeline_1DCNN.keras')
        output = np.argmax(model.predict([[[ 1.57912294],
       [ 1.32836936],
       [ 0.27590734],
       [ 1.05049255],
       [-1.14934812],
       [-0.30195909],
       [-0.28255704],
       [ 0.28507364],
       [-1.11445795],
       [ 0.08897337],
       [-0.28021359]]]), axis=1)

    return render_template('index.html', prediction_text='Wine Quality is $ {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
