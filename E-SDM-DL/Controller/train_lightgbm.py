

from Model.LightGBM import LightGBM
from Model.utils import Load

def train_lightgbm(sample, test, user, total_bit, SNR, input_type, num_boost_round, early_stopping_round):

    load = Load(sample, test, user, total_bit, SNR, input_type)
    input_data, input_test, answer_data, answer_test = load.load_data()

    model = LightGBM(input_data, input_test, answer_data, answer_test, num_boost_round, early_stopping_round)
    model.train()