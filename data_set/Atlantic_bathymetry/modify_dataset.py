import pandas as pd
import numpy as np

data = pd.read_csv("atlantic_data.txt", sep="\t")
data['long'] = data['long'].astype(float)
data['lat'] = data['lat'].astype(float)
data['depth'] = data['depth'].astype(float)
data.drop(data[data["lat"]>44.8].index, inplace=True)
data.drop(data[data["lat"]<44.5].index, inplace=True)
data.drop(data[data["long"]>-1.].index, inplace=True)
data.drop(data[data["long"]<-1.3].index, inplace=True)
print(data)
data.to_csv(r'atlantic_treated_data.csv', index=False, header=True)