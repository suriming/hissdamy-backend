from typing import Optional, Union
from fastapi import FastAPI
from mangum import Mangum
import pickle
import pandas as pd
from pydantic import BaseModel
import numpy as np
from sklearn.metrics import accuracy_score
import catboost
from fastapi.encoders import jsonable_encoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

app = FastAPI()

class KeypointsItem(BaseModel):
    x: float
    y: float
    z: float
    score:float
    name: str

class Scoringitem(BaseModel):
    keypoints: Union[list[KeypointsItem], None] = None

class Trainitem(BaseModel):
    frame: Union[list[Scoringitem], None] = None

with open('cat_weight.pkl', 'rb') as f:
    model = pickle.load(f)

@app.get("/")
async def root():
    return {"message": "CI/CD test"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

@app.post('/train/')
def train(item:Trainitem):
    
    return item


@app.post('/test/')
async def scoring_endpoint(item:Scoringitem):
    # df = pd.DataFrame(item.keypoints)
    # item = jsonable_encoder(item)
    data = pd.DataFrame(columns=['left_eye_inner_x', 'left_eye_x', 'left_eye_outer_x', 
                             'right_eye_inner_x', 'right_eye_x', 'right_eye_outer_x', 
                             'nose_x','mouth_left_x','mouth_right_x','ans'])

    data.loc[0] = [item.keypoints[1].x, item.keypoints[2].x, item.keypoints[3].x, 
                    item.keypoints[4].x, item.keypoints[5].x, item.keypoints[9].x,
                    item.keypoints[0].x, item.keypoints[9].x, item.keypoints[10].x, 0]
    # # print(df)
    # predict = model.predict(df)
    new_data = pd.DataFrame(columns=['0_2_dist_x', '0_5_dist_x', '0_2_5_diff_x', '1_4_dist_x', '0_9_10_diff_x', '0_9_10_ratio_x'] )
    new_data['0_2_dist_x'] = abs(data['left_eye_x'] - data['nose_x'])
    new_data['0_5_dist_x'] = abs(data['right_eye_x'] - data['nose_x'])
    new_data['0_2_5_diff_x'] = new_data['0_5_dist_x'] - new_data['0_2_dist_x']
    new_data['1_4_dist_x'] = abs(data['right_eye_inner_x'] - data['left_eye_inner_x'])
    new_data['0_9_10_diff_x'] = data['nose_x'] - ((data['mouth_left_x'] + data['mouth_right_x']) / 2)
    new_data['0_9_10_ratio_x'] = data['nose_x'] / ((data['mouth_left_x'] + data['mouth_right_x']) / 2)

    scaler = StandardScaler()
    # new_data = pd.DataFrame(new_data)
    scaler.fit(new_data)
    pred_X = scaler.transform(new_data)
    prediction = model.predict(pred_X)
    result = {}
    result['prediction'] = int(prediction)
    # ttt = jsonable_encoder(dict(prediction))  
    return result

@app.get("/predict")
def fetch_predictions(keypoints):

    return {}

handler = Mangum(app)