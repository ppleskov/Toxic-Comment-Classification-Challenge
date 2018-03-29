import sys, os, re, csv, codecs, numpy as np, pandas as pd
import spacy
from spacy import __version__
#print("Going to use Spacy version - ", __version__)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, concatenate, Conv1D
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

#EMBEDDING_FILE=f'{path}glove.42B.300d.txt' #glove.840B.300d glove.42B.300d glove.6B.300d
EMBEDDING_FILE=f'{path}crawl-300d-2M.vec' #wiki-news-300d-1M-subword wiki-news-300d-1M crawl-300d-2M

max_features = 300000 
embed_size = 300 

maxlen = 300
netlen = 500
batch_size = 350
epochs = 10
dropout = 0.2
num_filters = 10
patience = 3

repl = {
    "&lt;3": " good ",
    ":d": " good ",
    ":dd": " good ",
    ":p": " good ",
    "8)": " good ",
    ":-)": " good ",
    ":)": " good ",
    ";)": " good ",
    "(-:": " good ",
    "(:": " good ",
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " bad ",
    ":(": " bad ",
    ":s": " bad ",
    ":-s": " bad ",
    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bi'm\b": "i am",
    "m": "am",
    "r": "are",
    "u": "you",
    "haha": "ha",
    "hahaha": "ha",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "i'm": "i am",
    "m": "am",
    "i'll" : "i will",
    "its" : "it is",
    "it's" : "it is",
    "'s" : " is",
    "that's" : "that is",
    "weren't" : "were not",
}

def smile(word):
    if word in repl:
        return repl[word]
    return word

curse = {"fuck":"fuck",
         "suck":"suck",
         "cunt":"cunt", 
         "fuk":"fuck", 
         "crap":"crap", 
         "cock":"cock",
         "dick":"dick",
         "dumb":"dumb",
         "shit":"shit", 
         "bitch":"bitch", 
         "damn":"damn", 
         "piss":"piss", 
         "gay":"gay", 
         "fag":"faggot", 
         "assh":"asshole", 
         "basta":"bastard", 
         "douch":"douche",
         "haha":"haha",
         "nigger":"nigger",
         "penis":"penis",
         "vagina":"vagina",
         "niggors":"niggers",
         "nigors":"nigers",
         "fvckers":"fuckers",
         "phck":"fuck",
         "fack":"fuck",
         "sex":"sex",
         "wiki":"wikipedia",
         "viki":"wikipedia",
        }
def star(word):
    for x in fuck:
        if x in word:
            return fuck[x]
    return word

fuck = {"f**c":"fuck",        
        "f**k":"fuck",
        "f**i":"fuck",
        "f*ck":"fuck",
        "fu*k":"fuck",
        "shi*":"shit",
        "s**t":"shit",
        "sh*t":"shit",
        "f***":"fuck",
        "****i":"fuck",
        "c**t":"cunt",
        "b**ch":"bitch",
        "d**n":"damn",
        "*uck":"fuck",
        "fc*k":"fuck",
        "fu**":"fuck",
        "f*k":"fuck",
        "fuc*":"fuck",
        "f**":"fuck",
        }

#from emoji import UNICODE_EMOJI
#def is_emoji(s):
#    return s in UNICODE_EMOJI

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

TRAIN_DATA_FILE=f'{path}train.csv'
TEST_DATA_FILE=f'{path}test.csv'
PSUEDO_DATA_FILE = f'{path}2_avg_fast_glove_svm_toxic.csv'

train_all = pd.read_csv(TRAIN_DATA_FILE) #159571
test = pd.read_csv(TEST_DATA_FILE)
pseudo = pd.read_csv(PSUEDO_DATA_FILE)
#pseudo["comment_text"] = test["comment_text"]

print(len(pseudo))
for col in list_classes:
    pseudo.drop(pseudo[abs(pseudo[col]-0.5)<0.49].index, axis=0, inplace=True)
    print(len(pseudo))
for col in list_classes:
    pseudo[col] = [int(c) for c in pseudo[col]]
     
print('loading word embeddings...')
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

fname = EMBEDDING_FILE.split("/")[-1] + '.pkl'
if isfile(fname):
    with open(fname, 'rb') as handle:
        embeddings_index = pickle.load(handle)
else:
    #embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
    embeddings_index = {}
    f = codecs.open(EMBEDDING_FILE, encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        if len(coefs) == 300:
            embeddings_index[word] = coefs
        else:
            print(word)
    f.close()
    with open(fname, 'wb') as handle:
        pickle.dump(embeddings_index, handle, protocol=pickle.HIGHEST_PROTOCOL)



print('found %s word vectors' % len(embeddings_index))
max_features = min(max_features, len(embeddings_index)-1)

k = [0]*6
def magic(word):
    if word in embeddings_index:
        return word
    
    word = ''.join([i for i in word if not i.isdigit()])
    if word in embeddings_index:
        k[0]+=1
        return word
    
    new_word = "".join(OrderedDict.fromkeys(word))
    if new_word in embeddings_index:
        k[1]+=1
        return new_word
    
    for c in curse:
        if c in word:
            k[2]+=1
            return curse[c]
    t = 2
    if len(word)>=2*t:
        for i in range(t,len(word)-t+1):
            if (word[:i] in embeddings_index and
                word[i:] in embeddings_index):
                k[3]+=1
                return word[:i] + " " + word[i:]
    t = 2
    if len(word)>=3*t:
        for i in range(t,len(word)-2*t+1):
            for j in range(i+t,len(word)-t+1):
                if (word[:i] in embeddings_index and
                    word[i:j] in embeddings_index and
                    word[j:] in embeddings_index):
                    k[4]+=1
                    return word[:i] + " " + word[i:j] + " " + word[j:]
    
    if word!="":
        k[-1]+=1
        l.append(word)
    return word
                    
def preprocess(series):
    c = " "
    series = series.fillna("_na_").apply(lambda x: unidecode.unidecode(x).replace("’",c).replace("\'",c).replace("”",c).replace("“",c).replace("´",c).replace("‘",c).replace("•",c).replace("—"," ").replace("…",c).replace("·",c).replace("~",c).replace("`",c)).values 
    
    #series = [" ".join(smile(word.lower()) for word in line.split()) for line in series]
    #series = [" ".join(star(word.lower()) for word in line.split()) for line in series]
    
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    series = [" ".join(regex.sub(' ', line).lower().split()) for line in series]
    
    return np.array([" ".join(magic(word.lower()) for word in line.split()) for line in series])

lang = 'xx' #TODO: try en, lemma
nlp = spacy.load(lang, disable=['tagger', 'ner']) 

fname = f'text_train_{lang}.pkl'
if isfile(fname):
    with open(fname, 'rb') as handle:
        train_all["comment_text"] = pickle.load(handle)
else:
    tmp = []
    for line in tqdm(train_all["comment_text"].values):
        line = unidecode.unidecode(line.lower())
        try:
            doc = nlp(line)
            t = " ".join(token.text for token in doc)
            tmp.append(t)
        except:
            tmp.append(line)
            print(line)
    train_all["comment_text"] = tmp
    with open(fname, 'wb') as handle:
        pickle.dump(tmp, handle, protocol=pickle.HIGHEST_PROTOCOL)

fname = f'text_test_{lang}.pkl'
if isfile(fname):
    with open(fname, 'rb') as handle:
        test["comment_text"] = pickle.load(handle)
else:
    tmp = []
    for line in tqdm(test["comment_text"].values):
        line = unidecode.unidecode(line.lower())
        try:
            doc = nlp(line)
            t = " ".join(token.text for token in doc)
            tmp.append(t)
        except:
            tmp.append(line)
            print(line)
    test["comment_text"] = tmp
    with open(fname, 'wb') as handle:
        pickle.dump(tmp, handle, protocol=pickle.HIGHEST_PROTOCOL)

l = []
list_sentences_train_all = preprocess(train_all["comment_text"])
print(k); k = [0]*6
c = Counter(l)
#print(c.most_common(30))
with open("train.txt", "w") as f:
    f.write("")
with open("train.txt", "a") as f:
    for i,j in c.most_common():
        f.write(str(i)+"\t"+str(j)+"\n")

l = []
list_sentences_test = preprocess(test["comment_text"])
print(k)
c = Counter(l)
#print(c.most_common(30))
with open("test.txt", "w") as f:
    f.write("")
with open("test.txt", "a") as f:
    for i,j in c.most_common():
        f.write(str(i)+"\t"+str(j)+"\n")
del c
del l

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.scores = []
        self.interval = interval
        self.X_val, self.y_val = validation_data
    
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0, batch_size=1000)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f" % (epoch+1, score))
            self.scores.append(score)

def get_model(s):
    inp = Input(shape=(maxlen,), name="text")
    #num_vars = Input(shape=[s], name="num_vars")
    x = Embedding(min(max_features,nb_words), embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(dropout)(x)
    #x = Conv1D(embed_size, num_filters, activation="relu")(x)
    x = Bidirectional(GRU(netlen, return_sequences=True, dropout=dropout, recurrent_dropout=dropout))(x)
    #x = Conv1D(embed_size, num_filters, activation="relu")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dropout(dropout)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=[inp], outputs=x)
    
    #adam = Adam(beta_1=0.7, beta_2=0.99)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())
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
skf = KFold(n_splits=10, random_state=42, shuffle=True)

index = -1
scores = []
for train_index, valid_index in skf.split(range(len(train_all))):
    index+=1
    if index!=fold:
        continue
    #train = train_all.iloc[train_index,:]
    #valid = train_all.iloc[valid_index,:]
    
    iid = train_all.iloc[valid_index,:]["id"]

    y = train_all[list_classes].values
    y_train, y_valid = np.vstack((y[train_index,:], pseudo[list_classes].values)), y[valid_index,:]
    
    list_sentences_train = np.hstack((list_sentences_train_all[train_index], list_sentences_test[pseudo.index]))
    list_sentences_valid = list_sentences_train_all[valid_index]
    
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_valid = tokenizer.texts_to_sequences(list_sentences_valid)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_train = pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_valid = pad_sequences(list_tokenized_valid, maxlen=maxlen)
    X_test = pad_sequences(list_tokenized_test, maxlen=maxlen)
    
    X_tr = {"text":X_train, "num_vars":X_train_num[train_index,:]}
    X_va = {"text":X_valid, "num_vars":X_train_num[valid_index,:]}
    X_te = {"text":X_test,  "num_vars":X_test_num}
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    #print(emb_mean,emb_std)
    
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index)+1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: break
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector

    model = get_model(len(cols))
    
    filepath='./models/'+str(index)+'_lstm_'+str(dropout)+'.h5'
    #checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    #stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto')
    RocAuc = RocAucEvaluation(validation_data=(X_valid, y_valid), interval=1)
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=0, verbose=1)
    callbacks_list = [RocAuc]
    
    # exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
    # steps = int(len(X_train)/batch_size) * epochs
    # lr_init, lr_fin = 0.001, 0.0005
    # lr_decay = exp_decay(lr_init, lr_fin, steps)
    # K.set_value(model.optimizer.lr, lr_init)
    # K.set_value(model.optimizer.decay, lr_decay)
    if False:
        roc_score = 0
        lr = np.logspace(log(0.001), log(0.0001), base=e, num=epochs)
        for epoch in range(epochs):
            K.set_value(model.optimizer.lr, lr[epoch])
            print("Epoch:", epoch, "lr:", lr[epoch])
            model.fit(X_tr, y_train,
                      validation_data=(X_va, y_valid), 
                      batch_size=batch_size, 
                      epochs=1,
                      callbacks=callbacks_list
                     )

            if epoch==0:
                model.layers[1].trainable = True
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                print("Unfreeze them all")
            
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
        sample_submission.to_csv('./output/'+str(index)+'_lstm_test_'+str(dropout)+'_pseudo.csv', index=False)

        y_pred = model.predict([X_valid], batch_size=batch_size, verbose=1)
        df = pd.DataFrame({"id":iid})
        for c in list_classes:
            df[c] = [0]*len(df)
        df[list_classes] = y_pred
        df.to_csv('./output/'+str(index)+'_lstm_valid_'+str(dropout)+'_pseudo.csv', index=False)
    
    del model, X_train, X_valid, X_test, X_tr, X_va, X_te
    gc.collect()
if len(scores)>0:
    print(np.mean(scores),np.var(scores)*100000,scores)