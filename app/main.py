from typing import Optional, Union
from fastapi import FastAPI
from mangum import Mangum
# from matplotlib.pyplot import sca
import pandas as pd
from pydantic import BaseModel
import numpy as np
from catboost import CatBoostClassifier

from sklearn.preprocessing import StandardScaler
# from fastapi.middleware.cors import CORSMiddleware

from sklearn.model_selection import train_test_split
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import joblib
from sklearn.metrics import accuracy_score



app = FastAPI()

origins = [
    "http://localhost:3000", "http://0.0.0.0:3000", "http://127.0.0.1:8000", "http://localhost:3000/complete", "https://d2994vnof0kcde.cloudfront.net"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class KeypointsItem(BaseModel):
    x: float
    y: float
    z: float
    score:float
    name: str

class Scoringitem(BaseModel):
    keypoints: Union[list[KeypointsItem], None] = None

class Scoringitem_train(BaseModel):
    keypoints: Union[list[KeypointsItem], None] = None
    label: int

class Trainitem(BaseModel):
    frame: Union[list[Scoringitem_train], None] = None
    # frame: Union[Scoringitem, None] = None

# with open('cat_weight.pkl', 'rb') as f:
#     model = pickle.load(f)

@app.get("/")
async def root():
    return {"message": "CI/CD test"}

@app.post('/train/')
def train(item:Trainitem):
    data = pd.DataFrame(columns=['left_eye_inner_x', 'left_eye_x', 'left_eye_outer_x', 
                             'right_eye_inner_x', 'right_eye_x', 'right_eye_outer_x', 
                             'nose_x','mouth_left_x','mouth_right_x','ans'])
    for i in range(len(item.frame)):
        data.loc[i] = [item.frame[i].keypoints[1].x, item.frame[i].keypoints[2].x, item.frame[i].keypoints[3].x, 
                    item.frame[i].keypoints[4].x, item.frame[i].keypoints[5].x, item.frame[i].keypoints[9].x,
                    item.frame[i].keypoints[0].x, item.frame[i].keypoints[9].x, item.frame[i].keypoints[10].x, item.frame[i].label]

    new_data = pd.DataFrame(columns=['0_2_dist_x', '0_5_dist_x', '0_2_5_diff_x', '1_4_dist_x', '0_9_10_diff_x', '0_9_10_ratio_x'] )
    new_data['0_2_dist_x'] = abs(data['left_eye_x'] - data['nose_x'])
    new_data['0_5_dist_x'] = abs(data['right_eye_x'] - data['nose_x'])
    new_data['0_2_5_diff_x'] = new_data['0_5_dist_x'] - new_data['0_2_dist_x']
    new_data['1_4_dist_x'] = abs(data['right_eye_inner_x'] - data['left_eye_inner_x'])
    new_data['0_9_10_diff_x'] = data['nose_x'] - ((data['mouth_left_x'] + data['mouth_right_x']) / 2)
    new_data['0_9_10_ratio_x'] = data['nose_x'] / ((data['mouth_left_x'] + data['mouth_right_x']) / 2)
    new_data['ans'] = data['ans']

    X = new_data.drop(['ans'], axis=1)
    y = new_data[['ans']]
    print("FFF")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # scaler_filename = "scaler.pkl"
    # joblib.dump(scaler, scaler_filename)

    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    # X_val = scaler.transform(X_val)

    cat_clf = CatBoostClassifier(iterations=50,random_seed=42, max_bin=8)
    cat_clf.fit(X_train, y_train)
    y_pred = cat_clf.predict(X_val)

    filename = 'cat_weight.pkl'
    joblib.dump(cat_clf, filename)

    scr = accuracy_score(y_val, y_pred)

    return float(scr)


@app.post('/test/')
async def scoring_endpoint(item:Scoringitem):
    filename = 'cat_weight.pkl'
    model = joblib.load(filename)

    # with open('cat_weight.pkl', 'rb') as f:
    #     model = pickle.load(f)
    df = pd.DataFrame(item.keypoints)
    # item = jsonable_encoder(item)
    data = pd.DataFrame(columns=['left_eye_inner_x', 'left_eye_x', 'left_eye_outer_x', 
                             'right_eye_inner_x', 'right_eye_x', 'right_eye_outer_x', 
                             'nose_x','mouth_left_x','mouth_right_x'])

    data.loc[0] = [item.keypoints[1].x, item.keypoints[2].x, item.keypoints[3].x, 
                    item.keypoints[4].x, item.keypoints[5].x, item.keypoints[9].x,
                    item.keypoints[0].x, item.keypoints[9].x, item.keypoints[10].x]
    # # print(df)
    # predict = model.predict(df)
    new_data = pd.DataFrame(columns=['0_2_dist_x', '0_5_dist_x', '0_2_5_diff_x', '1_4_dist_x', '0_9_10_diff_x', '0_9_10_ratio_x'] )
    new_data['0_2_dist_x'] = abs(data['left_eye_x'] - data['nose_x'])
    new_data['0_5_dist_x'] = abs(data['right_eye_x'] - data['nose_x'])
    new_data['0_2_5_diff_x'] = new_data['0_5_dist_x'] - new_data['0_2_dist_x']
    new_data['1_4_dist_x'] = abs(data['right_eye_inner_x'] - data['left_eye_inner_x'])
    new_data['0_9_10_diff_x'] = data['nose_x'] - ((data['mouth_left_x'] + data['mouth_right_x']) / 2)
    new_data['0_9_10_ratio_x'] = data['nose_x'] / ((data['mouth_left_x'] + data['mouth_right_x']) / 2)

    # scaler_filename = "scaler.pkl"
    # scaler = joblib.load(scaler_filename)
    # new_data = pd.DataFrame(new_data)
    # scaler.fit(new_data)s
    # pred_X = scaler.transform(new_data)
    prediction = model.predict(new_data)
    # result = {}
    # result['prediction'] = float(prediction[0])
    # ttt = jsonable_encoder(dict(prediction))  
    return float(prediction[0])

handler = Mangum(app)