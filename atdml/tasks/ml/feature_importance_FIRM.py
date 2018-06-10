#! /usr/bin/python
'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''

# TBD for review/clean up
# python libraries
import os
import os.path
import re
import random
import sys, ConfigParser
import math
import numpy as np
import datetime
import zipfile
import shutil
import operator
from bson import json_util

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
from sklearn.externals import joblib

####pyspark#####
from pyspark import SparkContext
from pyspark.sql import SQLContext

#####matplotlib###############

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

####import our own library####
sys.path.append('./db')
import query_mongo
import ml_util
import zip_feature_util


####global constant
CONF_FILE='../../app.config' # at the base dir of the web
config=ConfigParser.ConfigParser()
config.read(CONF_FILE)
training_portion = eval(config.get("machine_learning","training_portion"))
mtx_name_list = config.get("machine_learning","mtx_name_list")
mtx_libsvm = config.get("machine_learning","mtx_libsvm")
mtx_stat = config.get("machine_learning","mtx_stat")
hdfs_file_name = 'libsvm_data'

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
    parser.add_argument("-f", "--folder", type=str, metavar="folder of features", help="hdfs folder contains features", required=False)
    parser.add_argument("-n", "--name", type=str, metavar="file name", help="file name for sample folder", required=False)
    parser.add_argument("-r", "--row_id", type=str, metavar="row_id number", help="row_id number in the db", required=False)
    
    parser.add_argument("-u", "--uploadtype", type=str, metavar="upload type", help="data type", required=False)
    parser.add_argument("-sc", "--scorefile", type=str, metavar="output score file", help="file name for output score", required=False)
    parser.add_argument("-o", "--out", type=str, metavar="output folder file", help="output folder file", required=False)
    parser.add_argument("-df", "--desc_file", type=str, metavar="feature description mapping file", help="feature description mapping file", required=False)
    parser.add_argument("-dsid", "--ds_id", type=str, metavar="source dataset id", help="source dataset id for training option", required=False)

    # Spark params            
    parser.add_argument('-sp','--sp_master', type=str, dest='sp_master', help='spark.master'
                , default =config.get('spark', 'spark_master'))
    parser.add_argument('-em','--exe_memory', type=str, dest='exe_memory', help='spark.executor.memory'
                , default =config.get('spark', 'spark_executor_memory'))
    parser.add_argument('-cm','--core_max', type=str, dest='core_max', help='spark.cores.max'
                , default =config.get('spark', 'spark_cores_max'))
    parser.add_argument("-tf", "--training_fraction", type=str, metavar="fraction for training set"
                , dest='training_fraction', help="fraction for training set", required=False)
    
    #### database
    parser.add_argument('-ip','--ip_address', type=str, dest='ip_address', help='mongodb ip address'
                , default =config.get('mongo', 'out_ip_address'))
    parser.add_argument('-p','--port', type=str, dest='port', help='mongodb port'
                , default =eval(config.get('mongo', 'out_port')))
    parser.add_argument('-dn','--db_name', type=str, dest='db_name', help='mongodb db name'
                , default =config.get('mongo', 'out_db'))
    parser.add_argument('-t','--tb_name', type=str, dest='tb_name', help='mongodb table name'
                , default =config.get('mongo', 'out_tb'))
    # auth
    parser.add_argument('-un','--username', type=str, dest='username', help='mongodb username'
                , default =config.get('mongo', 'out_username'))
    parser.add_argument('-pw','--password', type=str, dest='password', help='mongodb password'
                , default =config.get('mongo', 'out_password'))  
    parser.add_argument("-zo", "--zipout_dir", type=str, metavar="out figure folder", help="folder contains python code", required=False)
    parser.add_argument("-zc", "--zipcode_dir", type=str, metavar="out code", help="out code folder for python code", required=False)
    parser.add_argument("-zf", "--zipfilename", type=str, metavar="python code zip file", help="python code zip file for distribution to works", required=False)
                
                 
    args = parser.parse_args()
    
    if args.training_fraction:
        training_fraction = args.training_fraction
    else:
        training_fraction  = training_portion
    if args.folder:
        hdfs_feat_dir = args.folder
    else:
        hdfs_feat_dir  = config.get('app', 'HADOOP_MASTER')+'/user/hadoop/yigai/sality_virut_zbot_backdoor_hash_000'
    if args.name:
        file_name_given = args.name
    else:
        file_name_given  = 'aaaa'
    if args.row_id:
        row_id_str = args.row_id
    else:
        row_id_str  = '88'
    if args.ds_id:
        ds_id = args.ds_id
    else:
        ds_id  = ''
    if args.uploadtype:
        uploadtype = args.uploadtype
    else:
        uploadtype  = None
    if args.out:
        local_score_file = args.out
    else:
        local_score_file  = 'score_FIRM.txt'
    if len(args.username)>0:
        username = args.username
    else:
        username  = None
    if len(args.password)>0:
        password = args.password
    else:
        password  = None 

    if args.desc_file:
        description_file=args.desc_file
    else:
        description_file='./tbd.txt'
    if args.zipout_dir:
        zipout_dir = args.zipout_dir
    else:
        zipout_dir  = './ml'
    if args.zipcode_dir:
        zipcode_dir = args.zipcode_dir
    else:
        zipcode_dir  = './ml'
    if args.zipfilename:
        zip_file_name = args.zipfilename
    else:
        zip_file_name  = 'feat_impor_FIRM.zip'

    # mongo info for connection
    mongo_tuples=(args.ip_address, args.port, args.db_name, args.tb_name, username, password)
    
    return feat_importance_firm(row_id_str, ds_id, hdfs_feat_dir, local_score_file
    , args.sp_master,config.get('spark', 'spark_rdd_compress'),config.get('spark', 'spark_driver_maxResultSize'), args.exe_memory, args.core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples
    , training_fraction, 'feature_importance_FRIM:'+row_id_str,  uploadtype, description_file)


# ================================================================================== train () ============ 
def feat_importance_firm(row_id_str, ds_id, hdfs_feat_dir, local_score_file
    , sp_master, spark_rdd_compress, spark_driver_maxResultSize, sp_exe_memory, sp_core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples
    , training_fraction, jobname, uploadtype, description_file): 

    
    # zip func in other files for Spark workers ================= ================
    zip_file_path = ml_util.ml_build_zip_file(zipout_dir, zipcode_dir, zip_file_name, prefix='zip_feature_util')
    print "INFO: zip_file_path=",zip_file_path

    # get_spark_context
    sc=ml_util.ml_get_spark_context(sp_master
        , spark_rdd_compress
        , spark_driver_maxResultSize
        , sp_exe_memory
        , sp_core_max
        , jobname
        , [zip_file_path])
   
    
    t0 = time()
    
    # get feature seq mapping from mongo
    if uploadtype == "MD5 List IN-dynamic":
        ### connect to database to get the column list which contains all column number of the corresponding feature
        key = "dict_dynamic"
        jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
        jstr_proj='{"value":1}'

        # get parent dataset's data
        if ds_id != row_id_str:
            jstr_filter='{"rid":'+ds_id+',"key":"'+key+'"}'        
        
        doc=query_mongo.find_one_t(mongo_tuples, jstr_filter, jstr_proj)
        dic_list = doc['value']
        
        dic_all_columns = {}
        max_feature = 0
        # reverse dict{hashes:sequence number} ====== 
        for i in range(0, len(dic_list)):
            for key in dic_list[i]:
                dic_all_columns[eval(dic_list[i][key])] = key
                if eval(dic_list[i][key]) > max_feature:
                    max_feature = eval(dic_list[i][key])
        print "INFO: max_feature=",max_feature
        #print "dic_all_columns=",dic_all_columns # fid:numb,numb
    
    
    dirFile_loc = os.path.join(hdfs_feat_dir , "metadata")
    dirFolders = sc.textFile(dirFile_loc)
    
    hash_Folders = dirFolders.collect()
    #print "INFO: dirFile_loc=",dirFile_loc,", hash_Folders=",hash_Folders
    folder_list = [x.encode('UTF8') for x in hash_Folders]
    print "INFO: hdfs folder_list=",folder_list #['dirty/', 'clean/']
    
    # source libsvm filename  
    libsvm_data_file = os.path.join(hdfs_feat_dir , hdfs_file_name)
    print "INFO: libsvm_data_file=", libsvm_data_file
    
    # load feature count file
    #feat_count_file=libsvm_data_file+"_feat_count"
    #feature_count=zip_feature_util.get_feature_count(sc,feat_count_file)
    #print "INFO: feature_count=",feature_count

    # load sample RDD from text file   
    #samples_rdd, feature_count = zip_feature_util.get_sample_rdd(sc, libsvm_data_file, feature_count \
    #    , excluded_feat_cslist=None)
    samples_rdd=sc.textFile(libsvm_data_file).cache()

    # collect all data to local for processing ===============
    all_data = samples_rdd.collect()
    all_list = [ ln.split(' ') for ln in all_data ]
    sample_count=len(all_data)

    # label array
    #labels_list_all = [x.label for x,_ in all_data]
    #print "INFO: labels_list_all=",labels_list_all

    # get feature seq : ngram hash mapping ==================================
    key = "dic_seq_hashes"  #{"123":"136,345"}
    jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'

    jstr_proj='{"value":1}'
    
    # get parent dataset's data
    if ds_id != row_id_str:
        jstr_filter='{"rid":'+ds_id+',"key":"'+key+'"}'
            
    doc=query_mongo.find_one_t(mongo_tuples, jstr_filter, jstr_proj)
    dic_list = doc['value']
    
    dic_all_columns = dic_list
    feature_count = len(dic_list)

    # get hash : raw string mapping ==================================
    key = "dic_hash_str"  #{"123":"openFile"}
    jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
    jstr_proj='{"value":1}'
    # get parent dataset's data
    if ds_id != row_id_str:
        jstr_filter='{"rid":'+ds_id+',"key":"'+key+'"}'
        
    doc=query_mongo.find_one_t(mongo_tuples, jstr_filter, jstr_proj)
    dic_hash_str = doc['value']
    
    
    
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
    
    # loop through hdfs folders; TBD 
    for idx, folder in enumerate(folder_list):
        print "INFO: folder=", folder
        label = folder_list.index(folder) + 1
        print 'INFO: label=', label

        #logFile_name = os.path.join( hdfs_feat_dir, folder , mtx_name_list)
        #print "XXXXXXXXXXlogFile_name=",logFile_name
        #logFile_data = os.path.join( hdfs_feat_dir , folder , mtx_libsvm)
        #print "XXXXXXXXXXlogFile_data=",logFile_data

        '''
        logNames = sc.textFile(logFile_name).cache()
        logData = sc.textFile(logFile_data).cache()
        
        names = logNames.collect()
        data = logData.collect()
        
        name_l = [x.encode('UTF8') for x in names]
        feature_l = [x.encode('UTF8') for x in data]
        name_list = [names.strip() for names in name_l]
        feature_list = [features.strip() for features in feature_l]
        '''
        
        feature_list = [ l[2:] for l in all_list if int(l[1])==idx]
        # hash array
        name_list = [ l[2] for l in all_list if int(l[1])==idx ]
        #print "feature_list=",feature_list
        #print "name_list=",name_list
        
        ##########data seperation######
        id_perm = data_seperation_random(name_list)
        
        
        num_names = len(name_list)
        print 'INFO: num of samples=', num_names
        num_train = int(training_portion * num_names)
        print 'INFO: num_train = ', num_train

        
        ########generate training data#########
        i = 0;
        #print "INFO: generate training data"
        #print "INFO: len(id_perm)=",len(id_perm)
        while i < num_train:
            #print i, id_perm[i]
            features = feature_list[id_perm[i]]
            
            #features = features.strip()
            #feature_array = features.split(' ')
            feature_array=features
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
            
            
            ####for generating testing data folder####
            test_file_name = name_list[id_perm[i]]
            
  
            features = feature_list[id_perm[i]]

            #features = features.strip()
            #feature_array = features.split(' ')
            feature_array=features
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
    
    # end for loop here ========================
        
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
    #print "row_training:", row_training
    #print "INFO: col_training:", col_training
    len_col = len(col_training)
    print "INFO: col_num:", col_num
    labels_training = array(labels_training)

    features_testing = array(features_testing)
    row_testing = array(row_testing)

    col_testing = array(col_testing)
    labels_testing = array(labels_testing)

    
    sparse_mtx = csc_matrix((features_training,(row_training,col_training)), shape=(row_num_training,col_num))
    #print "sparse_mtx.todense(), sparse_mtx.shape=",sparse_mtx.todense(), sparse_mtx.shape
    
    sparse_test = csc_matrix((features_testing,(row_testing,col_testing)), shape=(row_num_testing,col_num))
    #print " sparse_test.todense(), sparse_test.shape=",sparse_test.todense(), sparse_test.shape
    
    clf = svm.LinearSVC()
    #clf = svm.SVC(C=0.1, kernel='rbf', degree=3, gamma=0.05, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
    #clf = svm.NuSVC(nu=0.3, kernel='rbf', degree=3, gamma=0.05, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False, max_iter=-1, random_state=None)
    #print "labels_training=",labels_training
    #print "sparse_mtx=",sparse_mtx
    clf.fit(sparse_mtx, labels_training)
    
    #print "INFO: model:intercept=",clf.intercept_
    #print "INFO: model:coef=",clf.coef_
    
    labels_pred = clf.predict(sparse_test)
    #print "labels_pred:", labels_pred
    
    accuracy = clf.score(sparse_test, labels_testing)
    #print "INFO: data folder=", hdfs_feat_dir
    print "INFO: accuracy=", accuracy
    
    #####################################################################
    ##################calculate feature importance with predication labels#######################
    #####################################################################
    AA = sparse_mtx.todense()
    BB = sparse_test.todense()
    labels_train_pred = clf.predict(sparse_mtx)
    labels_test_pred = labels_pred
    
    
    #print "###################################################################################"
    print "INFO: ======= Calculate feature importance with predication labels =================="
    #print "###################################################################################"
    dic_importance_label = {}
    
    for j in range (0, col_num):  ###for all features in the loop
    

        ##############################
        #print "====new way with sparse matrix========="
        curr_col_train = sparse_mtx.getcol(j)
        sum_col = curr_col_train.sum(0)
        positive_feature_number = int(sum_col.tolist()[0][0])
        
        labels_value = 3 - labels_train_pred
        dot_product = csr_matrix(np.array(labels_value)).dot(curr_col_train)
        sum_product = dot_product.sum(1)
        labels_positive_sum = int(sum_product.tolist()[0][0])        
        
        sum_label_values = sum(labels_value)
        labels_negitive_sum = sum_label_values - labels_positive_sum
        

        ##############################
        #print "====new way with sparse matrix========="
        curr_col_test = sparse_test.getcol(j)
        sum_col = curr_col_test.sum(0)
        positive_feature_number = positive_feature_number + int(sum_col.tolist()[0][0])
        
        labels_value = 3 - labels_test_pred
        dot_product = csr_matrix(np.array(labels_value)).dot(curr_col_test)
        sum_product = dot_product.sum(1)
        labels_positive_sum = labels_positive_sum + int(sum_product.tolist()[0][0])
        
        sum_label_values = sum(labels_value)
        labels_negitive_sum = labels_negitive_sum + sum_label_values - int(sum_product.tolist()[0][0])
        
        
        
        
        n_total = row_num_training + row_num_testing
        negitive_feature_number  = n_total - positive_feature_number
        if positive_feature_number == 0:
            #print "feature ", j+1, "all 0s!" 
            dic_importance_label[j+1] = -100
        elif negitive_feature_number == 0:
            #print "feature ", j+1, "all 1s!" 
            dic_importance_label[j+1] = -200
        else:
            q_positive = float(labels_positive_sum)/positive_feature_number
            q_negitive = float(labels_negitive_sum)/negitive_feature_number
        
            
            Q = (q_positive - q_negitive)*sqrt(float(q_positive)*q_negitive/float(n_total)/float(n_total))
            dic_importance_label[j+1] = Q
            
    
    sorted_importance = sorted(dic_importance_label.items(), key=operator.itemgetter(1), reverse=True)   
    print "INFO: ======= Feature Importance(FIRM score) ================"
    
    
    if os.path.exists(local_score_file):
        try:
            os.remove(local_score_file)
        except OSError, e:
            print ("ERROR: %s - %s." % (e.local_score_file,e.strerror))
    
    for ii in range (0, len(dic_importance_label)):
        (feat, score) = sorted_importance[ii]

        if dic_hash_str:
            description_str = feats2strs(dic_all_columns[str(feat)],dic_hash_str)
        else:
            description_str="N/A"
            print "Warning: No mapping found for feature number"
        
        str01 = str(feat)+"\t"+str(score)+"\t"+description_str+"\n"
        with open(local_score_file, "a") as f:
            f.write(str01)

    t1 = time()
    print 'INFO: running time: %f' %(t1-t0)
   
    print 'INFO: Finished!'
    return 0
    
# convert feature numb to string
def feats2strs(str_feats, dic_hash_str):
    ret=""
    comma=""
    #print "str_feats=",str_feats
    # get string from dic_hash_str
    for f in str_feats.split(','):
        if f in dic_hash_str:
            ret = ret + comma + dic_hash_str[f]
            comma=","
    return ret    
    
if __name__ == '__main__':
    __description__ = "feature importance: FIRM"
    main()
