import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import argparse
from typing import Tuple
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

DEFAULT_DATA_PATH=Path("data/milknew.csv")
DEFAULT_MODEL_PATH=Path("models/model.bin")
ETA_SELECTED = 0.03
MAX_DEPTH_SELECTED = 3
MIN_CHILD_WEIGHT_SELECTED = 1

@dataclass
class Dataset:
    df_full_train: pd.DataFrame
    y_full_train: np.ndarray


def load_prepare_data(path) -> pd.DataFrame:
    print("Loading data from:", path)
    df_raw = pd.read_csv(path)
    df = df_raw.copy()
    # clean column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df.rename(columns={"temprature": "temperature"}, inplace=True)
    df.rename(columns={"fat_": "fat"}, inplace=True)
    # Grade column values mapping
    grade_mapping = {"low": 0, "medium": 0, "high": 1}
    df["grade"] = df["grade"].map(grade_mapping)
    return df

def generate_dataset(df: pd.DataFrame) -> Dataset:
    print("Generating dataset...")
    df_full_train, _ = train_test_split(df, test_size=0.2, random_state=1)
    df_full_train = df_full_train.reset_index(drop=True)
    y_full_train = df_full_train['grade'].values
    del df_full_train['grade']
    dataset = Dataset(df_full_train, y_full_train)
    return dataset

def train_xgb_model(dataset: Dataset) -> Tuple[DictVectorizer, xgb.Booster]:
    print("Training XGBoost model...")
    dicts_full_train = dataset.df_full_train.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(dicts_full_train)
    features = list(dv.get_feature_names_out())
    dfulltrain = xgb.DMatrix(X_full_train, label=dataset.y_full_train, feature_names=features)

    xgb_params = {
        'eta': ETA_SELECTED,
        'max_depth': MAX_DEPTH_SELECTED,
        'min_child_weight': MIN_CHILD_WEIGHT_SELECTED,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 8,
        'seed': 1,
        'verbosity': 1,
    }

    modelXGB_final = xgb.train(xgb_params, dfulltrain, num_boost_round=75)
    return (dv, modelXGB_final)

def save_model(dv: DictVectorizer, model: xgb.Booster, path_to_save: str):
    print("Saving model...")
    with open(path_to_save, 'wb') as f_out:
        pickle.dump((dv, model), f_out)

def process_training(data_path: Path, model_path: Path):
    df = load_prepare_data(data_path)
    ds = generate_dataset(df)
    dv, model = train_xgb_model(ds)
    save_model(dv, model, model_path)
    print("Model saved to:", model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    process_training(args.data, args.model)