import joblib
import sklearn
import numpy

print("scikit-learn:", sklearn.__version__)
print("numpy:", numpy.__version__)

model = joblib.load("mental_health_model.pkl")

print("Model loaded successfully!")
print("Model type:", type(model))