import glob
import itertools
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn import preprocessing


ngram = [1]  # define the ngram and different type of nucleotide
nucl = ['A', 'C', 'G', 'T']  # define the nucleotides
data_tred = [0.5, 0.6, 0.7, 0.8, 0.85]  # define a list for the threshold for the DFF data range

# loading the dataset and modify them
csv_file = glob.glob("dff*.csv")[0]
df = pd.read_csv(csv_file, header=None)
df.columns = ["Sequence", "DFF"]  # defining the columns
df_drop = df.drop_duplicates(subset=['Sequence'])  # drop the duplicate sequences
nucl = ['A', 'C', 'G', 'T']  # define the nucleotides


def genComb(comb_len, nucl_list):
    fkey = {}
    keywords = [''.join(ga) for ga in itertools.product(nucl_list, repeat=comb_len)]
    for gb in keywords:
        fkey[gb] = 0
    return fkey


for tre in data_tred:
    # define the threshold and drop the data out of the threshold
    indexNames = df_drop[(df_drop['DFF'] < 0.9) & (df_drop['DFF'] > tre)].index
    df_tmp = df_drop.drop(indexNames)
    for ng in ngram:
        allcomb = {}
        result = []
        for seq in df_tmp['Sequence']:
            tmp = []
            for ls in range(0, len(seq)):
                if len(seq[ls:ls + ng]) == ng:
                    tmp.append(seq[ls:ls + ng])
            result.append(tmp)
            allcomb = genComb(ng, nucl)
        # encode all the combinations
        encoded_list = []
        for lst in result:
            tmp1 = []
            for i in lst:
                tmp1.append(list(allcomb).index(i))
            encoded_list.append(tmp1)
        # convert the list to array
        encoded_array = np.array(encoded_list)
        # encode the sequence to 0 and 1 and put in two
        onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
        onehot_encoder.fit(np.array(range(len(allcomb))).reshape(-1, 1))
        encoded_onehot = onehot_encoder.transform(encoded_array.reshape(-1, 1)).reshape(-1,
                                                                                        len(result[0]) * len(allcomb))
        X__ = encoded_onehot
        # convert to numpy array
        y__ = df_tmp['DFF'].values
        # = np_utils.to_categorical(Y)
        with open('pX' + str(int(tre * 100)) + '_' + str(ng) + 'g' + '.npy', 'wb') as f:
            np.save(f, X__)
        with open('py' + str(int(tre * 100)) + '_' + str(ng) + 'g' + '.npy', 'wb') as f:
            np.save(f, y__)
        df_tmp.to_csv(str(int(tre * 100)) + '_' + str(ng) + 'g' + '_m3.csv')
    df_tmp = 0
