from flask import Flask, render_template, redirect, url_for, request
import joblib

filename = 'svm_model_nm.joblib'
classifier = joblib.load(open(filename, 'rb'))
cv = joblib.load(open('tfidf_vectorizer.joblib', 'rb'))
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('Main.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        return redirect(url_for('predict'))

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        
        # # Convert prediction to labels
        if my_prediction[0] == 0:
            prediction_label = "Bad Tweet"
        else:
            prediction_label = "Good Tweet"
            
        return render_template('result.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
