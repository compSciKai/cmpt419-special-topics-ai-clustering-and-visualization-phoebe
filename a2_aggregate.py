# pylint: disable=invalid-name
# pylint: disable=missing-docstring

#import numpy as np
import glob
import pandas as pd

# parameters
p_path = "Phoebe_dataset4/Phoebe_dataset4/single_person_au_presence/train/"
i_path = "Phoebe_dataset4/Phoebe_dataset4/single_person_au_intensity/train/"
paths = [p_path, i_path]
conf_thresh = 0.75

# functions {{{
def preprocess(csv_path):

    # get data
    pre_df = pd.read_csv(csv_path)

    # formatting
    csv_name = str(csv_path).split("/")[-1]
    pre_df.columns = pre_df.columns.str.lstrip()
    pre_df.rename(columns={'Unnamed: 0':'frame'}, inplace=True)
    pre_df.insert(0, 'csv_name', csv_name)

    # preprocessing
    pre_df = pre_df[pre_df.success == 1]
    pre_df = pre_df[pre_df.confidence >= conf_thresh]

    return pre_df

# aggregate all data
def aggregate(path):

    data = []
    files = glob.glob(path + "/*.csv")
    for file in files:
        temp_df = preprocess(file)
        data.append(temp_df)

    df = pd.concat(data, axis=0, ignore_index=True)
    df = df.sort_values(['csv_name', 'frame'])

    return df

# }}}

# aggregate data
presence = aggregate(p_path)
intensity = aggregate(i_path)

# output
print(presence)
print(intensity)

# save csv
intensity.to_csv('intensity.csv', index=False)
presence.to_csv('presence.csv', index=False)
