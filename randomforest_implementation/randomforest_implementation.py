from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import utils
import constants

if __name__ == "__main__":
    data_array = utils.load_csv(constants.filename)
    X, y = utils.get_train_test_data(data_array)
    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    
    print(scores)

