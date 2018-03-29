import sys, os, re, csv, codecs, numpy as np, pandas as pd
import spacy
from spacy import __version__
#print("Going to use Spacy version - ", __version__)
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, concatenate, Conv1D, CuDNNLSTM, CuDNNGRU
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, SpatialDropout1D, GlobalMaxPooling1D
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.optimizers import Adam
from tqdm import tqdm
import pickle
from os.path import isfile
from collections import OrderedDict
import re
import unidecode
import string
from collections import Counter
import gc
from keras.callbacks import Callback
from keras import backend as K
from sklearn.metrics import roc_auc_score
import sys
from math import log, e
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

print("Fold:", sys.argv[1])
fold = int(sys.argv[1])

path = './input/'

EMBEDDING_FILE=f'{path}glove.6B.50d.txt' #glove.840B.300d glove.42B.300d glove.6B.300d
#EMBEDDING_FILE=f'{path}crawl-300d-2M.vec' #wiki-news-300d-1M-subword wiki-news-300d-1M crawl-300d-2M

max_features = 5000 
embed_size = 50

netlen = 100
batch_size = 100
epochs = 20
dropout = 0.0
min_lr = 0.0005
max_lr = 0.005

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

TRAIN_DATA_FILE=f'{path}train.csv'
TEST_DATA_FILE=f'{path}test.csv'
PSUEDO_DATA_FILE = f'{path}2_avg_fast_glove_svm_toxic.csv'

train_all = pd.read_csv(TRAIN_DATA_FILE) #159571
test = pd.read_csv(TEST_DATA_FILE)
pseudo = pd.read_csv(PSUEDO_DATA_FILE)
pseudo["comment_text"] = test["comment_text"]

# print(len(pseudo))
# for col in list_classes:
#     pseudo.drop(pseudo[abs(pseudo[col]-0.5)<0.49].index, axis=0, inplace=True)
# print(len(pseudo))
# for col in list_classes:
#     pseudo[col] = [int(c) for c in pseudo[col]]
     
def preprocess(series):
    c = ""
    series = series.fillna("_na_").apply(lambda x: unidecode.unidecode(x).lower()).values
    #.replace("’",c).replace("\'",c).replace("”",c).replace("“",c).replace("´",c).replace("‘",c).replace("•",c).replace("—"," ").replace("…",c).replace("·",c).replace("~",c).replace("`",c)).values 
    
    #regex = re.compile('[%s]' % re.escape(string.punctuation))
    #series = [" ".join(regex.sub(' ', line).lower().split()) for line in series]
    
    return np.array(series)
                    
list_sentences_train_all = preprocess(train_all["comment_text"])
list_sentences_test = preprocess(test["comment_text"])

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.scores = []
        self.interval = interval
        self.X_val, self.y_val = validation_data
    
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0, batch_size=batch_size*2)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f" % (epoch+1, score)) 
            self.scores.append(score)

def get_model(s):
    inp = Input(shape=(max_features,), name="text")
    #num_vars = Input(shape=[s], name="num_vars")
    x = Embedding(max_features, embed_size)(inp)
    #x = SpatialDropout1D(dropout)(x)
    #x = Conv1D(embed_size, num_filters, activation="relu")(x)
    x = Bidirectional(CuDNNGRU(netlen, return_sequences=True))(x)
    #x = Conv1D(embed_size, num_filters, activation="relu")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dropout(dropout)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=[inp], outputs=x)
    
    #adam = Adam(beta_1=0.7, beta_2=0.99)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

train_all['num_words'] = train_all.comment_text.str.count('\S+')
test['num_words'] = test.comment_text.str.count('\S+')
train_all['num_comas'] = train_all.comment_text.str.count('\.')
test['num_comas'] = test.comment_text.str.count('\.')
train_all['num_bangs'] = train_all.comment_text.str.count('\!')
test['num_bangs'] = test.comment_text.str.count('\!')
train_all['num_ques'] = train_all.comment_text.str.count('\?')
test['num_ques'] = test.comment_text.str.count('\?')
train_all['num_star'] = train_all.comment_text.str.count('\*')
test['num_star'] = test.comment_text.str.count('\*')
train_all['num_quotas'] = train_all.comment_text.str.count('\"')
test['num_quotas'] = test.comment_text.str.count('\"')
train_all['avg_word'] = train_all.comment_text.str.len() / (1 + train_all.num_words)
test['avg_word'] = test.comment_text.str.len() / (1 + test.num_words)

scaler = MinMaxScaler()
cols = ['num_words','num_comas','num_bangs','num_ques','num_quotas','avg_word','num_star']
X_train_num = scaler.fit_transform(train_all[cols])
X_test_num = scaler.transform(test[cols])

from sklearn.model_selection import StratifiedKFold, KFold
skf = KFold(n_splits=5, random_state=42, shuffle=True)

index = -1
scores = []
for train_index, valid_index in skf.split(range(len(train_all))):
    index+=1
    if index!=fold:
        continue
    
    iid = train_all.iloc[valid_index,:]["id"]

    y = train_all[list_classes].values
    y_train, y_valid = y[train_index,:], y[valid_index,:]
    #y_train, y_valid = np.vstack((y[train_index,:], pseudo[list_classes].values)), y[valid_index,:]
    #list_sentences_train = np.hstack((list_sentences_train_all[train_index], list_sentences_test[pseudo.index]))
    list_sentences_train = list_sentences_train_all[train_index]
    list_sentences_valid = list_sentences_train_all[valid_index]
    
    vec = TfidfVectorizer(ngram_range=(1,4),
                           max_features=max_features,
                           analyzer="char",
                           min_df=1,
                           max_df=0.9, 
                           strip_accents='unicode', 
                           use_idf=1,
                           smooth_idf=1, 
                           sublinear_tf=1)
    
    X_train = vec.fit_transform(list(list_sentences_train))
    X_valid = vec.transform(list(list_sentences_valid))
    X_test = vec.transform(list(list_sentences_test))
    
    print(len(vec.vocabulary_))
    
    #X_train = pad_sequences(list_tokenized_train, maxlen=maxlen)
    #X_valid = pad_sequences(list_tokenized_valid, maxlen=maxlen)
    #X_test = pad_sequences(list_tokenized_test, maxlen=maxlen)
    
    X_tr = {"text":X_train}#, "num_vars":X_train_num[train_index,:]}
    X_va = {"text":X_valid}#, "num_vars":X_train_num[valid_index,:]}
    X_te = {"text":X_test }#,  "num_vars":X_test_num}
    
    model = get_model(len(cols))
    
    filepath='./models/'+str(index)+'_lstm_'+str(dropout)+'.h5'
    RocAuc = RocAucEvaluation(validation_data=(X_valid, y_valid), interval=1)
    callbacks_list = [RocAuc]
    
    if True:
        roc_score = 0
        lr = np.logspace(log(max_lr), log(min_lr), base=e, num=epochs)
        for epoch in range(epochs):
            K.set_value(model.optimizer.lr, lr[epoch])
            print("Epoch:", epoch, "lr:", lr[epoch])
            model.fit(X_tr, y_train,
                      validation_data=(X_va, y_valid), 
                      batch_size=batch_size, 
                      epochs=1,
                      callbacks=callbacks_list
                     )

            if roc_score<RocAuc.scores[-1]:
                roc_score=RocAuc.scores[-1]
                model.save(filepath)
            else:
                scores.append(roc_score)
                break            
            
    else:
        model = load_model(filepath)

        y_test = model.predict([X_test], batch_size=batch_size, verbose=1)
        sample_submission = pd.read_csv(f'{path}sample_submission.csv')
        sample_submission[list_classes] = y_test
        sample_submission.to_csv('./output/'+str(index)+'_lstm_test_'+str(dropout)+'_char.csv', index=False)

        y_pred = model.predict([X_valid], batch_size=batch_size, verbose=1)
        df = pd.DataFrame({"id":iid})
        for c in list_classes:
            df[c] = [0]*len(df)
        df[list_classes] = y_pred
        df.to_csv('./output/'+str(index)+'_lstm_valid_'+str(dropout)+'_char.csv', index=False)
    
    del model, X_train, X_valid, X_test, X_tr, X_va, X_te
    gc.collect()
if len(scores)>0:
    print(np.mean(scores),np.var(scores)*100000,scores)