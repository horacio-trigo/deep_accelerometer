import pandas as pd
import os

def xpt_to_df(xpt_location, folder, name_xpt):
    df = pd.read_sas(os.path.join(xpt_location, folder, name_xpt))
    return df