#!/usr/bin/env python
# _*_coding:utf-8_*_
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score,cross_val_predict,GridSearchCV,StratifiedKFold,LeaveOneOut
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import auc,roc_auc_score,roc_curve,precision_recall_curve,confusion_matrix
from sklearn import metrics
from sklearn.utils import shuffle
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv1D,MaxPooling1D  #一维卷积层
from keras.layers import Dense,Flatten,Dropout,LSTM
from keras.optimizers import SGD,Adam,Adagrad,RMSprop
from keras.utils import np_utils,multi_gpu_model
import joblib
import os
import tensorflow as tf
from keras import backend as K
from sklearn import manifold, datasets, decomposition, ensemble,discriminant_analysis, random_projection, neighbors
from sklearn.preprocessing import StandardScaler,MinMaxScaler
np.set_printoptions(threshold=np.inf)
from sklearn.feature_selection import chi2,f_classif
from sklearn.feature_selection import SelectKBest
import sys,re
from keras.models import load_model
import optparse

def read_protein_sequences(testfile):
    with open(testfile) as f:
        records=f.read()
    check=0
    records = records.split('>')[1:]
    fasta_sequences = []
    sequence_name = []
    for fasta in records:
        segments = fasta.split('\n')
        name, sequence = segments[0].split()[0], re.sub('[^A-Z]', '', ''.join(segments[1:]).upper())
        if len(sequence)!=29:
            check=check+1
        fasta_sequences.append(([name, sequence]))
        sequence_name.append(str(name))
    return fasta_sequences, sequence_name,check



def AAindex_encoding(samples):
    with open('./AAindex/AAindex_normalized.txt') as f:
        records = f.readlines()[1:]
    AA_aaindex = 'ARNDCQEGHILKMFPSTWYV'
    AAindex = []
    AAindexName = []
    for i in records:
        AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
        AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)
    props = 'FINA910104:LEVM760101:JACR890101:ZIMJ680104:RADA880108:JANJ780101:CHOC760102:NADH010102:KYTJ820101:NAKH900110:GUYH850101:EISD860102:HUTJ700103:OLSK800101:JURD980101:FAUJ830101:OOBM770101:GARJ730101:ROSM880102:RICJ880113:KIDA850101:KLEP840101:FASG760103:WILM950103:WOLS870103:COWR900101:KRIW790101:AURR980116:NAKH920108'.split(':')
    if props:
        tmpIndexNames = []
        tmpIndex = []
        for p in props:
            if AAindexName.index(p) != -1:
                tmpIndexNames.append(p)
                tmpIndex.append(AAindex[AAindexName.index(p)])
        if len(tmpIndexNames) != 0:
            AAindexName = tmpIndexNames
            AAindex = tmpIndex

    index = {}
    for i in range(len(AA_aaindex)):
        index[AA_aaindex[i]] = i

    encoding_aaindex = []
    for i in samples:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        for aa in sequence:
            if aa == '-':
                for j in AAindex:
                    code.append(0)
                continue
            for j in AAindex:
                code.append(j[index[aa]])
        encoding_aaindex.append(code)
    return encoding_aaindex



def fea_write(name,encoding_aaindex):
    f=open(featurefile,'w')
    name=np.array(name).reshape(-1,1)
    numfea=len(encoding_aaindex[0])
    head0=['Fea_'+str(i+1) for i in range(numfea)]
    head=['Sequence_ID']+head0
    head=pd.DataFrame(columns=head)
    head.to_csv(f,index=False)
    encoding_aaindex=np.concatenate((name,encoding_aaindex),axis=1)
    encoding_aaindex=pd.DataFrame(encoding_aaindex)
    encoding_aaindex.to_csv(f,header=None,index=False)
    return encoding_aaindex


def col_delete(data):  #delete the columns woth same elements
    col_del=[406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434]
    data_new=np.delete(data,col_del,axis=1)
    return data_new

def create_model2():  #CNNs model
    model=Sequential()
    model.add(Conv1D(filters=128,kernel_size=5,input_shape=(top_feas,1),activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate=0.5))
    model.add(Conv1D(filters=128,kernel_size=5,activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate=0.5))
    model.add(Conv1D(filters=128,kernel_size=5,activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(1,activation='sigmoid'))
    #model = multi_gpu_model(model,gpus=4)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss=loss,optimizer=adam,metrics=[Sen,Spe,Accu,MCCs,Prec,F1score])
    return model

def ConfuMatrix(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    return TP,TN,FP,FN


def ConfuMatrix(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    return TP,TN,FP,FN
def Sen(y_true,y_pred):
    TP,TN,FP,FN=ConfuMatrix(y_true,y_pred)
    Sn = TP/(TP+FN)
    return Sn

def Spe(y_true,y_pred):
    TP,TN,FP,FN=ConfuMatrix(y_true,y_pred)
    Sp=TN/(TN+FP)
    return Sp

def Accu(y_true,y_pred):
    TP,TN,FP,FN=ConfuMatrix(y_true,y_pred)
    Acc = (TP+TN)/(TP+FN+FP+TN)
    return Acc

def MCCs(y_true,y_pred):
    TP,TN,FP,FN=ConfuMatrix(y_true,y_pred)
    MCC = ((TP*TN)-(FP*FN))/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)+0.00001)**0.5
    return MCC
def Prec(y_true,y_pred):
    TP,TN,FP,FN=ConfuMatrix(y_true,y_pred)
    Pre = TP/(TP+FP+0.00001)
    return Pre
def F1score(y_true,y_pred):
    TP,TN,FP,FN=ConfuMatrix(y_true,y_pred)
    Pre=TP/(TP+FP+0.00001)
    Re=TP/(TP+FN)
    F1=2*Pre*Re/(Pre+Re+0.00001)
    return F1


def binary_focal_loss(y_true, y_pred):
    # Define epsilon so that the backpropagation will not result in NaN
    # for 0 divisor case
    epsilon = K.epsilon()
    # Add the epsilon to prediction value
    #y_pred = y_pred + epsilon
    # Clip the prediciton value
    y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
    # Calculate p_t
    p_t = tf.where(K.equal(y_true, 1), y_pred, 1-y_pred)
    # Calculate alpha_t
    alpha_factor = K.ones_like(y_true)*alpha
    alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1-alpha_factor)
    # Calculate cross entropy
    cross_entropy = -K.log(p_t)
    weight = alpha_t * K.pow((1-p_t), gamma)
    # Calculate focal loss
    loss = weight * cross_entropy
    # Sum the losses in mini_batch
    loss = K.sum(loss, axis=1)
    return loss

        
def predict():
    if not os.path.exists(fastafile):
        print('Please give the protein sequence in FASTA format named as test.fasta!')
        exit()
    else:
        fastas, sequence_name,check = read_protein_sequences(fastafile)
        if check >0:
            print('Please check the input file:','\n','1> FASTA format','\n','2> window length=29')
            exit()
        else:
            encodings = AAindex_encoding(fastas)
            X = np.array(encodings).reshape(-1, 841)
            fea_write(sequence_name, X)  # write features to file
            X = col_delete(X)  # Delete features with the same value
            scale = joblib.load('./models/ss.pkl')
            X = scale.transform(X)
            X = np.expand_dims(X, axis=2)
            model = load_model(imodel,custom_objects={'binary_focal_loss':binary_focal_loss,'ConfuMatrix':ConfuMatrix,'Sen': Sen,'Spe':Spe,'Accu':Accu,'MCCs':MCCs,'Prec':Prec,'F1score':F1score})
            y_pred_proba = model.predict(X)
            df_out = pd.DataFrame(np.zeros((y_pred_proba.shape[0], 5)),columns=['Model', "ID", 'Sequence', "Scores", "Results"])
            for i in range(y_pred_proba.shape[0]):
                df_out.iloc[i, 0] = imodel[9:-3]
                df_out.iloc[i, 1] = str(sequence_name[i])
                df_out.iloc[i, 2] = fastas[i][1]
                df_out.iloc[i, 3] = "Kcr" if y_pred_proba[i] >= 0.5 else "non-Kcr"
                df_out.iloc[i, 4] = "%.2f%%" % (y_pred_proba[i] * 100)
            results = np.array(df_out)
            results_file = open(resultsfile, 'w')
            df_out.to_csv(results_file, index=None)
            results_file.close()
    return


if __name__=='__main__':
    thres=0.5
    alpha=8;gamma=0.25
    imodel = './models/CNNs_cpu.h5'
    parser = optparse.OptionParser()
    parser.add_option('-i', help='Fasta file name', dest='inputfile', action='store')
    parser.add_option('-o', help='Predicted results file', dest='outputfile', action='store')
    (opts, args) = parser.parse_args()
    fastafile=opts.inputfile
    featurefile=fastafile[:-6]+'_AAindex.csv'
    resultsfile=opts.outputfile
    predict()



