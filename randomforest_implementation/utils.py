import pandas as pd
import numpy as np

def load_csv(file_name):
    '''
    Load the CSV File into a pandas frame
    :param file_name:
    :return: Numpy data Frame
    '''
    df = pd.read_csv(file_name, header=0)
    df = df[pd.notnull(df['LOAD'])]
    data_array = df.as_matrix()
    # Remove first 3 columns: ZoneID, Timestamp
    data_array = np.delete(data_array, [0, 1], axis=1)
    return data_array


def get_train_test_data(data_array):
    return data_array[:, :1], data_array[:, 1:]
