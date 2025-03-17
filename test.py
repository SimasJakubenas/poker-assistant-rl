import os
import pandas as pd

folder_path = f"outputs/datasets/dataframe/v02"
if os.path.exists(folder_path): 
    data = pd.read_csv(f"{folder_path}/eval.csv")
    data['Total'] = data['3'].cumsum().round(2)
    print(data)