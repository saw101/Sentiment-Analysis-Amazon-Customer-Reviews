# Importing essential libraries
from flask import Flask, render_template, request
import pickle

# Load the Logistic Regression model and CountVectorizer object from local drive
filename = "Logistic_Reg_model.pkl"
classifier = pickle.load(open(filename, "rb"))
cv = pickle.load(open("countvector.pkl", "rb"))

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        Reviews = request.form["Reviews"]
        data = [Reviews]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template("result.html", prediction=my_prediction)


if __name__ == "__main__":
    app.run(debug=True)
