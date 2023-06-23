import pandas as pd
import numpy as np

data = pd.read_csv("./maritime_data.csv", sep=",").to_numpy()
data = data[data[:,1]<44.959459]
data = data[data[:,1]>44.400360]
data = data[data[:,0]<-0.927928] 
data = data[data[:,0]>-1.329730] 

data = pd.DataFrame(data)

data.to_csv(r'./maritime_data.csv', index=False, header=True)