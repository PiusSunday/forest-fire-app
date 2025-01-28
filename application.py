import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__, template_folder="src/templates")
app = application

## import ridge regressor model and standard scaler pickle
ridge_model = pickle.load(open("src/models/ridge.pkl", "rb"))
standard_scaler = pickle.load(open("src/models/scaler.pkl", "rb"))


## Route for home page
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictor", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        df = pd.DataFrame(
            [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        )

        new_data_scaled = standard_scaler.transform(df)

        result = ridge_model.predict(new_data_scaled)

        return render_template("predictor_page.html", result=result[0])

    else:
        return render_template("predictor_page.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090, debug=True)
