# import OS libraries
import glob
import os
# data preparing libraries
import numpy as np
import pandas as pd
# import keras libraries
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.layers.normalization import BatchNormalization
# plot librairs
import matplotlib.pyplot as plt
import seaborn as sns
# calculate parameters
from numpy import argmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, auc, roc_curve
# import cuda library to close the sessions
# from numba import cuda
import tensorflow as tf
from sklearn import preprocessing


predicting_list = glob.glob("*pre.csv")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

xnpy_f = glob.glob("pX*.npy")
ynpy_f = glob.glob("py*.npy")
xnpy_f.sort()
ynpy_f.sort()
lst_arr = [[xnpy_f[i], ynpy_f[i]] for i in range(len(xnpy_f))]

ngram = [1]

dict_para = {'data': [], 'random_state': [], 'ngram': [], 'c_cl0_train': [], 'c_cl1_train': [], 'c_cl0_test': [],
             'c_cl1_test': [], 'accuracy': [], 'precission_0': [], 'precission_1': [], 'recall_0': [], 'recall_1': [],
             'f1_0': [], 'f1_1': [], 'AUC_0': [], 'AUC_1': [], 'TN': [], 'FP': [], 'FN': [], 'TP': []}

for ar in lst_arr:
    # preparing the data
    psv_fname = 'psv' + str(ar[0][2:4])  # make a dir psv names
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
                                                                , stratify=y
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
                if int(y_train[ctr][0]) == 1:
                    count_train_class_0 = count_train_class_0 + 1
                else:
                    count_train_class_1 = count_train_class_1 + 1

            for cte in range(len(y_test)):
                if int(y_test[cte][0]) == 1:
                    count_test_class_0 = count_test_class_0 + 1
                else:
                    count_test_class_1 = count_test_class_1 + 1

            model = Sequential()
            model.add(Conv2D(2048, (3, 3), input_shape=(X_train.shape[1], X_train.shape[2], 1), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(1, 1)))
            model.add(Conv2D(2048, (3, 3), padding='same'))
            model.add(BatchNormalization(axis=-1))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(1, 1)))
            model.add(Flatten())
            model.add(Dense(2048))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.2))
            model.add(Dense(2))
            model.add(Activation('sigmoid'))
            model.compile(
                loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )
            history = model.fit(
                X_train, y_train,
                batch_size=128,
                epochs=20,
                validation_split=0.2,
                verbose=False,
                shuffle=True
            )
            # saving the plot and csv of loss and accuracy
            fig = plt.figure()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            fig.savefig(subrn_fname + '/loss.png')
            df_loss = pd.DataFrame(history.history['loss'])
            df_loss.columns = ['loss']
            df_loss['val_loss'] = history.history['val_loss']
            df_loss.to_csv(subrn_fname + '/loss.csv', index=False)
            fig = plt.figure()
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.grid(True)
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            fig.savefig(subrn_fname + '/acc.png')
            df_acc = pd.DataFrame(history.history['accuracy'])
            df_acc.columns = ['accuracy']
            df_acc['val_accuracy'] = history.history['val_accuracy']
            df_acc.to_csv(subrn_fname + '/accuracy.csv', index=False)

            dict_test = {'test_Num': [], 'Pred_R': [], 'Exp_R': [], 'PoF': [], 'P_cl0': [], 'P_cl1': []}
            for tseq in range(len(X_test)):
                y_pred = model.predict(X_test[tseq].reshape(1, X_test.shape[1], X_test.shape[2], 1))
                y_class = model.predict_classes(X_test[tseq].reshape(1, X_test.shape[1], X_test.shape[2], 1))
                dict_test['test_Num'].append(tseq)
                dict_test['P_cl0'].append(y_pred[0][0])
                dict_test['P_cl1'].append(y_pred[0][1])
                dict_test['Pred_R'].append(int(y_class))
                if int(y_test[tseq][0]) == 1:
                    dict_test['Exp_R'].append(0)
                elif int(y_test[tseq][1]) == 1:
                    dict_test['Exp_R'].append(1)
                if dict_test['Exp_R'][tseq] == dict_test['Pred_R'][tseq]:
                    dict_test['PoF'].append('Pass')
                else:
                    dict_test['PoF'].append('Fail')
            # save the pridicted to the pandas datafram
            df_predicted = pd.DataFrame.from_dict(dict_test)
            df_predicted.to_csv(subrn_fname + '/pred_xtest.csv', index=False)
            # calculate parameters
            y_prob = model.predict_proba(X_test)
            y_test_decode = argmax(y_test, axis=1)
            # auc for class zero polt and csv save
            fpr0, tpr0, _ = roc_curve(y_test_decode, y_prob[:, 0], pos_label=0)
            roc_auc_0 = auc(fpr0, tpr0)
            fig = plt.figure()
            plt.plot([0, 1], [0, 1], '--')
            plt.plot(fpr0, tpr0)
            plt.xlabel("FPR-0")
            plt.ylabel("TPR-0")
            plt.title("AUC-0 : {:.2f}".format(roc_auc_0))
            fig.savefig(subrn_fname + '/auc0.png')
            df_auc0 = pd.DataFrame(fpr0)
            df_auc0.columns = ["FPR-0"]
            df_auc0["TPR-0"] = tpr0.tolist()
            df_auc0.to_csv(subrn_fname + '/class0.csv', index=False)
            # auc for class one polt and csv save
            fpr1, tpr1, _ = roc_curve(y_test_decode, y_prob[:, 1], pos_label=1)
            roc_auc_1 = auc(fpr1, tpr1)
            fig = plt.figure()
            plt.plot([0, 1], [0, 1], '--')
            plt.plot(fpr1, tpr1)
            plt.xlabel("FPR-1")
            plt.ylabel("TPR-1")
            plt.title("AUC-1 : {:.2f}".format(roc_auc_1))
            fig.savefig(subrn_fname + '/auc1.png')
            df_auc1 = pd.DataFrame(fpr1)
            df_auc1.columns = ["FPR-1"]
            df_auc1["TPR-1"] = tpr1.tolist()
            df_auc1.to_csv(subrn_fname + '/class1.csv', index=False)
            # calclate f1-score, recall
            y_predict = model.predict_classes(X_test)
            fscore_0 = f1_score(y_test_decode, y_predict, pos_label=0)
            fscore_1 = f1_score(y_test_decode, y_predict, pos_label=1)
            precision_0 = precision_score(y_test_decode, y_predict, pos_label=0)
            precision_1 = precision_score(y_test_decode, y_predict, pos_label=1)
            recall_0 = recall_score(y_test_decode, y_predict, pos_label=0)
            recall_1 = recall_score(y_test_decode, y_predict, pos_label=1)
            acc_sco = accuracy_score(y_test_decode, y_predict)
            # True negative ...
            con_mat = confusion_matrix(y_test_decode, y_predict)
            tn = con_mat[0][0]
            fp = con_mat[0][1]
            fn = con_mat[1][0]
            tp = con_mat[1][1]
            fig = plt.figure()
            sns_plot = sns.heatmap(con_mat, annot=True, cmap='coolwarm')
            fig.savefig(subrn_fname + '/tnfp.png')
            if fscore_0 >= 0.6 and fscore_1 >= 0.6:
                model.save(subrn_fname + '/' + psv_fname + '_' + str(ng) + 'gram' + '_' + str(dr) + 'rnds' + '.h5')
            dict_para['data'].append(psv_fname)
            dict_para['ngram'].append(ng)
            dict_para['random_state'].append(dr)
            dict_para['c_cl0_train'].append(count_train_class_0)
            dict_para['c_cl1_train'].append(count_train_class_1)
            dict_para['c_cl0_test'].append(count_test_class_0)
            dict_para['c_cl1_test'].append(count_test_class_1)
            dict_para['accuracy'].append(acc_sco)
            dict_para['precission_0'].append(precision_0)
            dict_para['precission_1'].append(precision_1)
            dict_para['recall_0'].append(recall_0)
            dict_para['recall_1'].append(recall_1)
            dict_para['f1_0'].append(fscore_0)
            dict_para['f1_1'].append(fscore_1)
            dict_para['AUC_0'].append(roc_auc_0)
            dict_para['AUC_1'].append(roc_auc_1)
            dict_para['TN'].append(tn)
            dict_para['FP'].append(fp)
            dict_para['FN'].append(fn)
            dict_para['TP'].append(tp)
            if fscore_0 >= 0.6 and fscore_1 >= 0.6:
                for prcsv in predicting_list:
                    df_predicting = pd.read_csv(prcsv, header=None)
                    df_predicting.columns = ["Sequence"]
                    Xpr = df_predicting['Sequence'].values
                    Xpr_ = np.array([list(_) for _ in Xpr])
                    le = preprocessing.LabelEncoder().fit(['A', 'C', 'G', 'T'])
                    Xpr_ = le.transform(Xpr_.reshape(-1)).reshape(-1, 18)

                    onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
                    onehot_encoder.fit(np.array([0, 1, 2, 3]).reshape(-1, 1))
                    Xpr__ = onehot_encoder.transform(Xpr_.reshape(-1, 1)).reshape(-1, 18, 4, 1)
                    dict_expr_test = {'Test Number': [], 'Sequence': [], 'Predicted_Reads': [], 'P_cl0': [],
                                      'P_cl1': [], 'Result': []}
                    for pri in range(len(Xpr__)):
                        y_pred_xpr = model.predict(Xpr__[pri].reshape(1, 18, 4, 1))
                        y_class_xpr = model.predict_classes(Xpr__[pri].reshape(1, 18, 4, 1))
                        dict_expr_test['Test Number'].append(pri)
                        dict_expr_test['P_cl0'].append(y_pred_xpr[0][0])
                        dict_expr_test['P_cl1'].append(y_pred_xpr[0][1])
                        dict_expr_test['Predicted_Reads'].append(int(y_class_xpr))
                        dict_expr_test['Sequence'].append(Xpr[pri])
                        if int(y_class_xpr) == 1:
                            dict_expr_test['Result'].append('BIND')
                        elif int(y_class_xpr) == 0:
                            dict_expr_test['Result'].append('NO-BIND')
                    df_expr_test = pd.DataFrame.from_dict(dict_expr_test)
                    df_expr_test.to_csv(subrn_fname + '/' + psv_fname + '_' + str(prcsv[:-4]) + str(dr) + 'rnds' +
                                        '.csv', index=False)

            plt.close('all')
            tf.keras.backend.clear_session()
            model = 0
            history = 0
#             cuda.select_device(0)
#             cuda.close()

df_para = pd.DataFrame.from_dict(dict_para)
df_para.to_csv('fresult.csv')
