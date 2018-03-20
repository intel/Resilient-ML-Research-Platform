#! /usr/bin/python
'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''

# python libraries
import os
import os.path
import re
import random
import sys, ConfigParser
import math
import numpy as np
import datetime

from argparse import ArgumentParser
from scipy.sparse import *
from scipy import *
from time import time
from sklearn import svm, grid_search, datasets
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.svm import NuSVC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import RandomizedPCA

####pyspark#####
from pyspark import SparkContext
from pyspark.sql import SQLContext

#####matplotlib###############
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

#####import for django database####
sys.path.append('./db')
import exec_sqlite

####global constant
CONF_FILE='../../app.config' # at the base dir of the web
config=ConfigParser.ConfigParser()
config.read(CONF_FILE)
portion = eval(config.get("machine_learning","portion"))
mtx_name_list = config.get("machine_learning","mtx_name_list")
mtx_libsvm = config.get("machine_learning","mtx_libsvm")
mtx_stat = config.get("machine_learning","mtx_stat")

def data_seperation_date(name_l):
    date_list = []
    for name in name_l:
        f_without_ext, ext = os.path.splitext(name)
        d1,d2,d3,fname = f_without_ext.split('_', 3)
        date = int(float(d1))*10000 + int(float(d2))*100 + int(float(d3))
        date_list.append(date)
    
    a = np.array(date_list)
    id_perm = np.argsort(a).tolist()
    return id_perm 

def data_seperation_random(names):
    num = len(names)
    id_perm = range(num)
    random.shuffle(id_perm)
    return id_perm

def main():
    
    parser = ArgumentParser(description=__description__)
    parser.add_argument("-f", "--folder", type=str, metavar="folder of features", help="folder contains features", required=False)
    parser.add_argument("-n", "--name", type=str, metavar="file name", help="file name for sample folder", required=False)
    parser.add_argument("-o", "--out", type=str, metavar="out figure folder", help="folder contains output", required=False)
    parser.add_argument("-r", "--row_id", type=str, metavar="row_id number", help="row_id number in the table", required=False)
    parser.add_argument("-b", "--bin", type=str, metavar="bin number", help="number of bins for var plot", required=False)
    parser.add_argument("-mn", "--run", type=str, metavar="run number", help="number of runs for var plot", required=False)
    parser.add_argument("-u", "--uploadtype", type=str, metavar="upload type", help="data type", required=False)
    parser.add_argument("-w", "--fromweb", type=str, metavar="flag for web", help="flag for web", required=False)
    parser.add_argument('-sp','--sp_master', type=str, dest='sp_master', help='spark.master'
                , default =config.get('spark', 'spark_master'))
    parser.add_argument('-em','--exe_memory', type=str, dest='exe_memory', help='spark.executor.memory'
                , default =config.get('spark', 'spark_executor_memory'))
    parser.add_argument('-cm','--core_max', type=str, dest='core_max', help='spark.cores.max'
                , default =config.get('spark', 'spark_cores_max'))
    args = parser.parse_args()
    
    if args.folder:
        feat_dir = args.folder
    else:
        feat_dir  = config.get('app', 'HADOOP_MASTER')+'/user/hadoop/yigai/sality_virut_zbot_backdoor_dic_000'
    if args.name:
        file_name_given = args.name
    else:
        file_name_given  = 'bbbb'
    if args.out:
        out_dir = args.out
    else:
        out_dir  = 'out_result'
    if args.row_id:
        row_id_str = args.row_id
    else:
        row_id_str  = '1'
    if args.bin:
        bin_number = eval(args.bin)
    else:
        bin_number  = 10
    if args.run:
        run_number = eval(args.run)
    else:
        run_number  = 2

    if args.uploadtype:
        uploadtype = args.uploadtype
    else:
        uploadtype  = None
    if args.fromweb:
        fromweb = args.fromweb
    else:
        fromweb  = None
    
    data_folder = feat_dir + "/"
    out_dir = out_dir + "/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    SparkContext.setSystemProperty('spark.rdd.compress', config.get('spark', 'spark_rdd_compress'))
    SparkContext.setSystemProperty('spark.driver.maxResultSize', config.get('spark', 'spark_driver_maxResultSize'))
    #SparkContext.setSystemProperty('spark.kryoserializer.buffer.mb', config.get('spark', 'spark_kryoserializer_buffer_mb'))
    SparkContext.setSystemProperty('spark.executor.memory', args.exe_memory)
    SparkContext.setSystemProperty('spark.cores.max', args.core_max)
    
    sc = SparkContext(args.sp_master, 'multi_run:'+str(args.row_id))
    
    
    dirFile_loc = data_folder + "metadata"
    dirFolders = sc.textFile(dirFile_loc)
    
    hash_Folders = dirFolders.collect()
    print hash_Folders
    folder_list = [x.encode('UTF8') for x in hash_Folders]
    print folder_list
    
    features_training = []
    labels_training = []
    names_training = []
    row_training = []
    col_training = []
    max_feat_training = 0
    row_num_training = 0
    features_testing = []
    labels_testing = []
    names_testing = []
    row_testing = []
    col_testing = []
    max_feat_testing = 0
    row_num_testing = 0
    for folder in folder_list:
        print "****folder:", folder
        logFile_name = data_folder + folder + mtx_name_list

        logFile_data = data_folder + folder + mtx_libsvm

        logNames = sc.textFile(logFile_name).cache()
        logData = sc.textFile(logFile_data).cache()
        
        names = logNames.collect()
        data = logData.collect()
        name_l = [x.encode('UTF8') for x in names]
        feature_l = [x.encode('UTF8') for x in data]
        name_list = [names.strip() for names in name_l]
        feature_list = [features.strip() for features in feature_l]

        
        ##########data seperation######
        id_perm = data_seperation_random(name_list)

        
        num_names = len(name_list)
        print 'num of samples in ', logFile_data, ' = ', num_names
        num_train = int(portion * num_names)
        print 'num_train = ', num_train
        label = folder_list.index(folder) + 1
        print 'labe of ', logFile_data, ' is ', label
        
        ########generate training data#########
        i = 0;
        print "here"
        print len(id_perm)
        while i < num_train:
            #print i, id_perm[i]
            features = feature_list[id_perm[i]]
            
            features = features.strip()
            feature_array = features.split(' ')
            labels_training.append(label)
            
            length = len(feature_array)
            j = 0
            while j < length:
                feature = feature_array[j]
                feat, value = feature.split(':', 2)
                row_training.append(i + row_num_training)
                col_training.append(int(feat) - 1)
                features_training.append(int(value))
                max_feat_training = max(max_feat_training, int(feat))
                j = j+1
            i = i+1
        row_num_training = row_num_training + num_train
        i = num_train
        ########generate testing data#########
        while i < num_names:
            #print i, id_perm[i]
            features = feature_list[id_perm[i]]

            features = features.strip()
            feature_array = features.split(' ')
            labels_testing.append(label)
            
            length = len(feature_array)
            j = 0
            while j < length:
                feature = feature_array[j]
                feat, value = feature.split(':', 2)
                row_testing.append(i - num_train + row_num_testing)
                col_testing.append(int(feat) - 1)
                features_testing.append(int(value))
                max_feat_testing = max(max_feat_testing, int(feat))
                j = j+1
            i = i+1
        row_num_testing = row_num_testing + (num_names - num_train)
        
    col_num = max(max_feat_training, max_feat_testing)
    if max_feat_training < col_num:
        for i in range (0, row_num_training):
            for j in range(max_feat_training, col_num):
                features_training.append(0)
                row_training.append(i)
                col_training.append(j)
    elif max_feat_testing < col_num:
        for i in range (0, row_num_testing):
            for j in range(max_feat_testing, col_num):
                features_testing.append(0)
                row_testing.append(i)
                col_testing.append(j)

    features_training = array(features_training)
    row_training = array(row_training)
    col_training = array(col_training)
    print "col_training:", col_training
    len_col = len(col_training)
    for ii in range(0, len_col):
        if col_training[ii] < 0:
            print "=======!! < 0 ====== index:", ii
            print "value: ", col_training[ii]
    print "col_num:", col_num
    labels_training = array(labels_training)

    features_testing = array(features_testing)
    row_testing = array(row_testing)
    col_testing = array(col_testing)
    labels_testing = array(labels_testing)

    
    print "***************"
    print features_training[0].shape, features_testing[0].shape
    
    sparse_mtx = csr_matrix((features_training,(row_training,col_training)), shape=(row_num_training,col_num))
    #print sparse_mtx.todense(), sparse_mtx.shape
    
    sparse_test = csr_matrix((features_testing,(row_testing,col_testing)), shape=(row_num_testing,col_num))
    #print sparse_test.todense(), sparse_test.shape
    
    clf = svm.LinearSVC()
    #clf = svm.SVC(C=0.1, kernel='rbf', degree=3, gamma=0.05, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
    #clf = svm.NuSVC(nu=0.3, kernel='rbf', degree=3, gamma=0.05, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False, max_iter=-1, random_state=None)
    clf.fit(sparse_mtx, labels_training)
    labels_pred = clf.predict(sparse_test)
    print "************results*********"
    print "Predicting results:"
    print labels_pred
    print "True testing labels:"
    print labels_testing
    
    with open("result.txt", 'w') as f:
        f.write("prediction vs true\n")
    num_err = 0
    for i in range (0, len(labels_pred)):
        if labels_pred[i] != labels_testing[i]:
            num_err = num_err + 1
        with open("result.txt", 'a') as f:
            f.write('%d  ' % (labels_pred[i]))
            f.write('%d\n' % (labels_testing[i]))
    print "correct percentage : ", 1-float(num_err)/len(labels_pred)
    
    print labels_pred.shape, labels_testing.shape
    
    accuracy = clf.score(sparse_test, labels_testing)
    print "data folder:", data_folder
    print "accuracy: ", accuracy
        
    #######################################################################
    #########plot accuracy variance and distribution########
    t0 = time()
    num_run = run_number   ###50 is default
    accuracy_array = np.zeros(num_run)
    for rnd in range (0, num_run):
        
        dirFile_loc = data_folder + "metadata"
        dirFolders = sc.textFile(dirFile_loc)
        
        hash_Folders = dirFolders.collect()
        
        folder_list = [x.encode('UTF8') for x in hash_Folders]
        
        
        features_training = []
        labels_training = []
        names_training = []
        row_training = []
        col_training = []
        max_feat_training = 0
        row_num_training = 0
        features_testing = []
        labels_testing = []
        names_testing = []
        row_testing = []
        col_testing = []
        max_feat_testing = 0
        row_num_testing = 0
        for folder in folder_list:
            
            logFile_name = data_folder + folder + mtx_name_list
            
            logFile_data = data_folder + folder + mtx_libsvm
            
            
            logNames = sc.textFile(logFile_name).cache()
            logData = sc.textFile(logFile_data).cache()
            
            names = logNames.collect()
            data = logData.collect()
            name_l = [x.encode('UTF8') for x in names]
            feature_l = [x.encode('UTF8') for x in data]
            name_list = [names.strip() for names in name_l]
            feature_list = [features.strip() for features in feature_l]
            
            
            ##########data separation######
            id_perm = data_seperation_random(name_list)
            
            
            num_names = len(name_list)
            
            num_train = int(portion * num_names)
            
            label = folder_list.index(folder) + 1
            
            
            ########generate training data#########
            i = 0;
            
            while i < num_train:
                
                features = feature_list[id_perm[i]]
                
                features = features.strip()
                feature_array = features.split(' ')
                labels_training.append(label)
                
                length = len(feature_array)
                j = 0
                while j < length:
                    feature = feature_array[j]
                    feat, value = feature.split(':', 2)
                    row_training.append(i + row_num_training)
                    col_training.append(int(feat) - 1)
                    features_training.append(int(value))
                    max_feat_training = max(max_feat_training, int(feat))
                    j = j+1
                i = i+1
            row_num_training = row_num_training + num_train
            i = num_train
            ########generate testing data#########
            while i < num_names:
                
                features = feature_list[id_perm[i]]

                features = features.strip()
                feature_array = features.split(' ')
                labels_testing.append(label)
                
                length = len(feature_array)
                j = 0
                while j < length:
                    feature = feature_array[j]
                    feat, value = feature.split(':', 2)
                    row_testing.append(i - num_train + row_num_testing)
                    col_testing.append(int(feat) - 1)
                    features_testing.append(int(value))
                    max_feat_testing = max(max_feat_testing, int(feat))
                    j = j+1
                i = i+1
            row_num_testing = row_num_testing + (num_names - num_train)
            
        col_num = max(max_feat_training, max_feat_testing)
        if max_feat_training < col_num:
            for i in range (0, row_num_training):
                for j in range(max_feat_training, col_num):
                    features_training.append(0)
                    row_training.append(i)
                    col_training.append(j)
        elif max_feat_testing < col_num:
            for i in range (0, row_num_testing):
                for j in range(max_feat_testing, col_num):
                    features_testing.append(0)
                    row_testing.append(i)
                    col_testing.append(j)

        features_training = array(features_training)
        row_training = array(row_training)
        col_training = array(col_training)

        len_col = len(col_training)

        labels_training = array(labels_training)

        features_testing = array(features_testing)
        row_testing = array(row_testing)
        col_testing = array(col_testing)
        labels_testing = array(labels_testing)

        
        
        sparse_mtx = csr_matrix((features_training,(row_training,col_training)), shape=(row_num_training,col_num))

        
        sparse_test = csr_matrix((features_testing,(row_testing,col_testing)), shape=(row_num_testing,col_num))

        
        clf = svm.LinearSVC()
        #clf = svm.SVC(C=0.1, kernel='rbf', degree=3, gamma=0.05, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
        #clf = svm.NuSVC(nu=0.3, kernel='rbf', degree=3, gamma=0.05, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False, max_iter=-1, random_state=None)
        clf.fit(sparse_mtx, labels_training)
        labels_pred = clf.predict(sparse_test)
        
        with open("result.txt", 'w') as f:
            f.write("prediction vs true\n")
        num_err = 0
        for i in range (0, len(labels_pred)):
            if labels_pred[i] != labels_testing[i]:
                num_err = num_err + 1
            with open("result.txt", 'a') as f:
                f.write('%d  ' % (labels_pred[i]))
                f.write('%d\n' % (labels_testing[i]))

        
        accuracy = clf.score(sparse_test, labels_testing)
        accuracy_array[rnd] = accuracy

        print "current round: ", rnd
    
    #######plot distribution and variance#####
    plt.figure(1)
    
    num_bins = bin_number  ####10 is default
    n, bins, patches = plt.hist(accuracy_array, num_bins, normed=1, facecolor='green', alpha=0.5)
    ave = np.mean(accuracy_array)
    print "Accuracy mean: ", ave
    variance = np.std(accuracy_array)
    print "Accuracy variance: ", variance
    
    print "bins: ", bins
    # add a 'best fit' line
    y = mlab.normpdf(bins, ave, variance)
    print "y: ", y
    plt.plot(bins, y, 'r--')
    
    plt.title('Accuracy distribution of '+str(num_run)+' runs:')
    plt.xlabel('Accuracy Values')
    plt.ylabel('Probability')
    
    plt.savefig(out_dir+file_name_given+"_var_"+str(num_run)+".png")
    

    t1 = time()
    print 'running time: %f' %(t1-t0)
    

    # only update db for web request
    if fromweb=="1": 
        #print "database update"
        str_sql="UPDATE atdml_document set "+"mean = '"+str(ave*100)+"%"+"', variance = '"+str(variance*100) \
            +"%',status = 'mruned', processed_date ='"+str(datetime.datetime.now()) \
            +"' where id="+row_id_str
        ret=exec_sqlite.exec_sql(str_sql)
        print "Data update done! ret=", str(ret)
    else:
        print "mean = '"+str(mean*100)+"%"
        print "variance = '"+str(variance*100)+"%"



    print 'Finished!'

if __name__ == '__main__':
    __description__ = "multiple run for plotting variance"
    main()
