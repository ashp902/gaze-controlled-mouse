from platform import architecture
from unicodedata import name
from keras import models, layers
import pandas as pd

data = pd.read_csv('data.csv')

print(data)

n_features = 154

model = models.Sequential(name = 'DeppNN', layers =[
    layers.Dense(name='hidden1', input_shape=(n_features,), units = int(round((n_features + 2) / 2)), activation='relu'),
    layers.Dropout(name='drop1', rate=0.2),
    layers.Dense(name='hidden2', units = int(round((n_features + 2) / 4)), activation='relu'),
    layers.Dropout(name='drop2', rate=0.2),
    layers.Dense(name='hidden3', units = int(round((n_features + 2) / 8)), activation='relu'),
    layers.Dense(name='output', units=2, activation='sigmoid'),
])

model.build()
model.summary()