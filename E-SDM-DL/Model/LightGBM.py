
import lightgbm as lgb
import pandas as pd
import numpy as np

from optuna.integration import lightgbm_tuner
from sklearn.metrics import accuracy_score

class LightGBM:
    def __init__(self, input_data, input_test, answer_data, answer_test, num_boost_round, early_stopping_round):
        self.X_train = pd.DataFrame(input_data)
        self.X_test = pd.DataFrame(input_test)

        self.ans_len = answer_data.shape[1]

        ans_data = np.zeros((answer_data.shape[0], 1), dtype=np.int)
        ans_test = np.zeros((answer_test.shape[0], 1), dtype=np.int)

        for i in range(answer_data.shape[0]):
            ans_data[i, 0] = list(answer_data[i, :]).index(1)

        for i in range(answer_test.shape[0]):
            ans_test[i, 0] = list(answer_test[i, :]).index(1)

        self.y_train = pd.DataFrame(ans_data)
        self.y_test = pd.DataFrame(ans_test)

        self.num_boost_round = num_boost_round
        self.early_stopping_round = early_stopping_round


    def train(self):

        lgb_train = lgb.Dataset(self.X_train, self.y_train)
        lgb_eval = lgb.Dataset(self.X_test, self.y_test)

        params = {'objective': 'multiclass',
                  'num_class': self.ans_len,
                  'metric': 'multi_error'}

        best_params = {}
        tuning_history = []

        model = lightgbm_tuner.train(params,
                                     lgb_train,
                                     valid_sets=lgb_eval,
                                     num_boost_round=self.num_boost_round,
                                     early_stopping_rounds=self.early_stopping_round,
                                     verbose_eval=False,
                                     best_params=best_params,
                                     tuning_history=tuning_history,
                                     )

        self.y_pred_prob = model.predict(self.X_test, num_iteration=model.best_iteration)
        self.y_pred = np.argmax(self.y_pred_prob, axis=1)

        df_pred = pd.DataFrame({'target':self.y_test[0], 'target_pred':self.y_pred})
        print(df_pred)

        # df_pred_prob = pd.DataFrame({'y':self.y_test[0],
        #                              'target0_prob':self.y_pred_prob[:,0], 'target1_prob':self.y_pred_prob[:,1], 'target2_prob':self.y_pred_prob[:,2],
        #                              'target3_prob':self.y_pred_prob[:,3], 'target4_prob':self.y_pred_prob[:,4], 'target5_prob':self.y_pred_prob[:,5],
        #                              'target6_prob':self.y_pred_prob[:,6], 'target7_prob':self.y_pred_prob[:,7], 'target8_prob':self.y_pred_prob[:,8],
        #                              'target9_prob':self.y_pred_prob[:,9], 'target10_prob':self.y_pred_prob[:,10], 'target11_prob':self.y_pred_prob[:,11],
        #                              'target12_prob':self.y_pred_prob[:,12]})

        acc = accuracy_score(self.y_test, self.y_pred)
        print('Acc :', acc)

