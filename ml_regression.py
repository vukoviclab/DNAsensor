from sklearn.svm import SVR
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
# import OS libraries
import glob
import os
import joblib
# data preparing libraries
import numpy as np
import pandas as pd
# import keras libraries
from keras.utils import np_utils
# plot librairs
import matplotlib.pyplot as plt
# calculate parameters
from sklearn.model_selection import train_test_split

xnpy_f = glob.glob("pX*.npy")
ynpy_f = glob.glob("py*.npy")
xnpy_f.sort()
ynpy_f.sort()
lst_arr = [[xnpy_f[i], ynpy_f[i]] for i in range(len(xnpy_f))]
predicting_list = glob.glob("*pre.csv")

model_names = ["SVM_Linear", "SVM_RBF", "SVM_Sigmoid"]
classifiers = [

    SVR(kernel='linear'),
    SVR(kernel='rbf'),
    SVR(kernel='sigmoid'),
]
scoring = 'accuracy'

ngram = [1]
dict_para = {'model_name': [], 'data': [], 'random_state': [], 'ngram': [], 'c_cl0_train': [], 'c_cl1_train': [],
             'c_cl0_test': [], 'c_cl1_test': [],
             'r2_score': [], 'mean_squared_error': [], 'mean_absolute_error': [],
             'max_error': [], 'explained_variance_score': []}

diff_models = zip(model_names, classifiers)
results_ml = []
model_names = []
for name, model in diff_models:
    model_fname = str(name)
    os.makedirs(model_fname)
    for ar in lst_arr:
        # preparing the data
        psv_fname = model_fname + '/psv' + str(ar[0][2:4])  # make a dir psv names
        psv_alone = 'psv' + str(ar[0][2:4])
        os.makedirs(psv_fname)
        for ng in ngram:
            subngr_fname = psv_fname + '/' + str(ng) + 'gram'  # make a subdir ngrams
            os.makedirs(subngr_fname)
            # selsct the ngram from the df cloumn name and put it to a list
            X = np.load(ar[0])
            y = np.load(ar[1])
            for dr in range(1, 201):
                subrn_fname = subngr_fname + '/' + 'rnst' + str(dr).zfill(3)  # make a subdir random state
                os.makedirs(subrn_fname)
                X_train, X_test, y_train, y_test = train_test_split(X, y
                                                                    , random_state=dr
                                                                    )
                with open(subrn_fname + '/X_train_' + str(ng) + 'gram' + '.npy', 'wb') as f:
                    np.save(f, X_train)
                with open(subrn_fname + '/X_test_' + str(ng) + 'gram' + '.npy', 'wb') as f:
                    np.save(f, X_test)
                with open(subrn_fname + '/y_train_' + str(ng) + 'gram' + '.npy', 'wb') as f:
                    np.save(f, y_train)
                with open(subrn_fname + '/y_test_' + str(ng) + 'gram' + '.npy', 'wb') as f:
                    np.save(f, y_test)
                count_train_class_0 = 0
                count_train_class_1 = 0
                count_test_class_0 = 0
                count_test_class_1 = 0

                for ctr in range(len(y_train)):
                    if int(y_train[ctr]) == 1:
                        count_train_class_0 = count_train_class_0 + 1
                    else:
                        count_train_class_1 = count_train_class_1 + 1

                for cte in range(len(y_test)):
                    if int(y_test[cte]) == 1:
                        count_test_class_0 = count_test_class_0 + 1
                    else:
                        count_test_class_1 = count_test_class_1 + 1

                kfold = model_selection.KFold(n_splits=10)
                model.fit(X_train, y_train)

                dict_test = {'test_Num': [], 'Pred_R': [], 'Exp_R': []}
                for tseq in range(len(X_test)):
                    y_reg = model.predict(X_test[tseq].reshape(1, -1))
                    dict_test['test_Num'].append(tseq)
                    dict_test['Pred_R'].append(y_reg[0])
                    dict_test['Exp_R'].append(y_test[tseq])

                # save the pridicted to the pandas datafram
                df_predicted = pd.DataFrame.from_dict(dict_test)
                df_predicted.to_csv(subrn_fname + '/pred_xtest.csv', index=False)
                x_fig_plt = df_predicted['Pred_R'].values
                y_fig_plt = df_predicted['Exp_R'].values
                fig_predict_plt = plt.figure()
                plt.scatter(x_fig_plt, y_fig_plt)
                plt.xlabel('Pred_R')
                plt.ylabel('Exp_R')
                plt.plot([min(x_fig_plt), max(y_fig_plt)], [min(x_fig_plt), max(y_fig_plt)], '--', c='red')
                fig_predict_plt.savefig(subrn_fname + '/pred_xtest.png', dpi=600)
                # calculate parameters
                y_predict = model.predict(X_test)
                
                joblib.dump(model, subrn_fname + '/' + psv_alone + '_' + str(ng) + 'gram' + '_' + str(dr) + 'rnds' +
                            '.sav')

                m_r2_score = metrics.r2_score(y_test, y_predict)
                m_mean_squared_error = metrics.mean_squared_error(y_test, y_predict)
                m_mean_absolute_error = metrics.mean_absolute_error(y_test, y_predict)
                m_max_error = metrics.max_error(y_test, y_predict)
                m_explained_variance_score = metrics.explained_variance_score(y_test, y_predict)
                dict_para['model_name'].append(str(name))
                dict_para['data'].append(psv_fname)
                dict_para['ngram'].append(ng)
                dict_para['random_state'].append(dr)
                dict_para['c_cl0_train'].append(count_train_class_0)
                dict_para['c_cl1_train'].append(count_train_class_1)
                dict_para['c_cl0_test'].append(count_test_class_0)
                dict_para['c_cl1_test'].append(count_test_class_1)
                dict_para['r2_score'].append(m_r2_score)
                dict_para['mean_squared_error'].append(m_mean_squared_error)
                dict_para['mean_absolute_error'].append(m_mean_absolute_error)
                dict_para['max_error'].append(m_max_error)
                dict_para['explained_variance_score'].append(m_explained_variance_score)
                # predicting the last sequences
                for prcsv in predicting_list:
                    df_predicting = pd.read_csv(prcsv, header=None)
                    df_predicting.columns = ["Sequence"]
                    Xpr = df_predicting['Sequence'].values
                    Xpr_ = np.array([list(_) for _ in Xpr])
                    le = preprocessing.LabelEncoder().fit(['A', 'C', 'G', 'T'])
                    Xpr_ = le.transform(Xpr_.reshape(-1)).reshape(-1, 18)
                    onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
                    onehot_encoder.fit(np.array([0, 1, 2, 3]).reshape(-1, 1))
                    Xpr__ = onehot_encoder.transform(Xpr_.reshape(-1, 1)).reshape(-1, len(X_train[0]))
                    dict_expr_test = {'Test Number': [], 'Sequence': [], 'Predicted_Reads': []}
                    for pri in range(len(Xpr__)):
                        y_class_xpr = model.predict(Xpr__[pri].reshape(1, -1))
                        dict_expr_test['Test Number'].append(pri)
                        dict_expr_test['Predicted_Reads'].append(int(y_class_xpr))
                        dict_expr_test['Sequence'].append(Xpr[pri])
                    df_expr_test = pd.DataFrame.from_dict(dict_expr_test)
                    df_expr_test.to_csv(
                        subrn_fname + '/' + psv_alone + '_' + str(ng) + 'gram' + '_' + str(dr) + 'rnds_' +
                        str(prcsv[:-4]) + '.csv', index=False)

                plt.close('all')

df_para = pd.DataFrame.from_dict(dict_para)
df_para.to_csv('fresult.csv')
