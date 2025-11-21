import joblib
import numpy as np

# load your model
model = joblib.load("my_model.pkl")

print("Model type:", type(model))

# show what classes the model knows (most sklearn models have this)
if hasattr(model, "classes_"):
    print("Model classes_:", model.classes_)
else:
    print("Model has no classes_ attribute")

# run one dummy prediction using the same feature order you trained on:
# [age, sbp, spo2, hr]
X = np.array([[40, 120, 98, 80]])
pred = model.predict(X)

print("Prediction output:", pred)
print("Prediction data type:", type(pred[0]))
