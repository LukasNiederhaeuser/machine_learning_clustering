import os
import pandas as pd

# define folder
FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def read_file(folder: str, filename: str, delimiter: str) -> pd.DataFrame:
        
    if folder == "raw":
        data = pd.read_csv(os.path.join(FOLDER, folder, filename), delimiter=delimiter)
    elif folder == "processed":
        data = pd.read_csv(os.path.join(FOLDER, folder, filename), delimiter=delimiter)
    else:
        print("Folder can be either: 'raw' or 'processed'")
        
    return data