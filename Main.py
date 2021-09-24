import pandas as pd
import numpy as np

data = pd.read_csv("Concrete_Compressive_Strength_data_set.csv")
random = data.sample(frac = 1).reset_index(drop=True)
print(data)
print(random)
matrix = random.to_numpy()



