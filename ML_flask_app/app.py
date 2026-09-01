from flask import Flask, render_template, request
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import os
import time

app = Flask(__name__)

# Load trained model, its feature order, and the SHAP explainer
model = pickle.load(open("model.pkl", "rb"))
FEATURE_NAMES = pickle.load(open("feature_names.pkl", "rb"))
explainer = pickle.load(open("explainer.pkl", "rb"))

LOCAL_SHAP_PATH = os.path.join("static", "shap_local.png")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
    except ValueError:
        return render_template(
            "index.html",
            prediction_text="⚠️ Please enter valid numbers in all fields.",
        )

    final_features = np.array(features).reshape(1, -1)

    prediction = model.predict(final_features)

    # ---- Explainable AI: why did the model predict this value? ----
    shap_values_single = explainer.shap_values(final_features)

    # For some sklearn/shap version combos, expected_value comes back as a
    # 1-element array instead of a plain float. shap.Explanation needs a
    # scalar base value, so unwrap it if needed.
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[0]

    # Build a SHAP Explanation object so we can use shap.plots.waterfall
    explanation = shap.Explanation(
        values=shap_values_single[0],
        base_values=base_value,
        data=final_features[0],
        feature_names=FEATURE_NAMES,
    )

    plt.figure()
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(LOCAL_SHAP_PATH, dpi=150)
    plt.close()

    return render_template(
        "index.html",
        prediction_text=f"Predicted CO Level: {prediction[0]:.3f}",
        show_explanation=True,
        cache_buster=int(time.time()),
    )


if __name__ == "__main__":
    # use_reloader=False: the app writes a new PNG (static/shap_local.png)
    # into the watched project folder on every prediction. Flask's default
    # auto-reloader treats that as a code change and restarts the server
    # mid-request, which drops the connection ("connection failed" in the
    # browser). Keeping debug=True still gives full error pages if something
    # goes wrong, it just won't auto-restart on file writes.
    app.run(debug=True, use_reloader=False)