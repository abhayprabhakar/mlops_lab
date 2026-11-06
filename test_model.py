import joblib
import numpy as np
def test_model_prediction():
    model = joblib.load('diabetes_model.joblib')
    input_data = np.array(
        [0.02, 0.07, 0.03, 0.02, 0.01,
         0.03, 0.01, 0.04, 0.03, 0.01]
    ).reshape(1, -1)
    prediction = model.predict(input_data)
    assert isinstance(prediction[0], (float, np.floating)), \
        f"Expected float but got {type(prediction[0])}"


def test_model_output_shape():
    model = joblib.load('diabetes_model.joblib')
    input_data = np.zeros((5, 10))
    predictions = model.predict(input_data)
    assert predictions.shape == (5,), \
        f"Expected shape (5,) but got {predictions.shape}"