import numpy as np
from flask import Flask, request, render_template
import joblib
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    if request.method == 'POST':
        text = request.form['Review']
        data = [text]
        vectorizer = cv.transform(data).toarray()
        prediction = model.predict(vectorizer)

        if prediction:
            return render_template('index.html', review_text=text, prediction_text='The review is Positive')
            # return ("The review is POSITVIE")
        else:
            return render_template('index.html', review_text=text, prediction_text='The review is Negative.')
    else:
        return render_template('index.html')  # Add this to handle GET requests


if __name__ == "__main__":
    app.run(debug=True)