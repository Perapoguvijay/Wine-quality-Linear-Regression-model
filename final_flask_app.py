from flask import Flask, render_template, request, jsonify
import joblib

model = joblib.load(r"C:\Users\vijay\OneDrive\Desktop\Datafiles of nareshit\lr_wine_model.joblib")

app = Flask(__name__)

@app.route("/")
def greet():
    return render_template("index.html")

@app.route("/predict", methods=["GET"])
def predict():
    try:
        flavonoids = request.args.get("Falvonoida", type=float)
        Malic_Acid = request.args.get("MalicAcid", type=float)
        Intencity = request.args.get("Color_intensity", type=float)
        Alcohol = request.args.get("Alcohol", type=float)
        Proline = request.args.get("Proline", type=float)

        features = [flavonoids, Malic_Acid, Intencity, Alcohol, Proline]
        result = model.predict([features])
        return jsonify(f"The result is: {result[0]}")
    except Exception as e:
        return jsonify(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
