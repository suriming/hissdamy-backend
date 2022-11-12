import pandas as pd
from pydantic import BaseModel
import joblib

# class ssdamModel:

#     def __init__(self):
#         self.model_fname = 'cat_weight.pkl'
#         try:
#             self.model = joblib.load(self.model_fname)
#         except Exception as _:
#             self.model = self._train_model()
#             joblib.dump(self.model, self.model_fname_)

#     def _train_model(self):
#         X = self.df.
