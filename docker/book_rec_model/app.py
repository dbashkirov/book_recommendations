import os
import lightfm
from fastapi import FastAPI
import mlflow
from dotenv import load_dotenv
import pickle
import scipy.sparse as sp
from typing import List
import numpy as np

load_dotenv()

app = FastAPI()

remote_server_uri = "http://62.217.183.172:5000"
mlflow.set_tracking_uri(remote_server_uri)


class Model:
    def __init__(self, run_id, dst_path="models"):
        self.artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=dst_path)
        with open(f"{self.artifact_path}/model.pkl", "rb") as f:
            self.model = pickle.load(f)
        self.item_features = sp.load_npz(f"{self.artifact_path}/book_features.npz")

    def predict(self, user_id, item_id):
        return self.model.predict(user_id, item_id, item_features=self.item_features)


model = Model("3d9f30d8eb2642bfbc3b5e1d66b03bf1")


@app.post("/predict")
async def predict(user_ids: List[int], item_ids: List[int]):
    score = model.predict(np.array(user_ids), np.array(item_ids))
    return list(zip(user_ids, item_ids, score.tolist()))
