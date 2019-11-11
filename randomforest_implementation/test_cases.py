import utils
import constants
import numpy as np

def load_csv_file():
    data_loaded = utils.load_csv(constants.filename)
    print(data_loaded)


if __name__ == "__main__":
    data_loaded = utils.load_csv(constants.filename)
    X, y = utils.get_train_test_data(data_loaded)
    print("yo")