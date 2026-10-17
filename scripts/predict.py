import pickle
import argparse
import xgboost as xgb
from pydantic import BaseModel

from pathlib import Path
from dataclasses import dataclass
from sklearn.feature_extraction import DictVectorizer

DEFAULT_MODEL_PATH=Path("models/modelXB.bin")

class Milk(BaseModel):
    ph: float
    temperature: float
    taste: int
    odor: int
    fat: int
    turbidity: int
    colour: int

class PredictionResponse(BaseModel):
    prediction: float
    is_high_grade: bool

def load_model(path: Path) -> tuple[DictVectorizer, xgb.Booster]:
    print("Loading model from:", path)
    with open(path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return (dv, model)

def predict(dv: DictVectorizer, model: xgb.Booster, milk: Milk) -> int:
    print("Predicting...")
    X = dv.transform(milk.model_dump())
    features = list(dv.get_feature_names_out())
    dX = xgb.DMatrix(X, feature_names=features)
    return round(model.predict(dX)[0], 3)

def process_prediction(model_path: Path, milk: Milk) -> PredictionResponse:
    dv, model = load_model(model_path)
    grade_proba = predict(dv, model, milk)
    selected_threshold = 0.4
    is_high_grade = grade_proba >= selected_threshold
    return PredictionResponse(prediction=grade_proba, is_high_grade=is_high_grade)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict quality grade")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    milk = Milk(ph=6.8, temperature=45, taste=0, odor=1, fat=1, turbidity=1, colour=255)
    prediction = process_prediction(args.model, milk)
    print('sample')
    print('-----')
    print(milk)
    print('-----')
    print(f'prediction = {prediction.prediction}')
    print(f'Is High Grade = {prediction.is_high_grade}')