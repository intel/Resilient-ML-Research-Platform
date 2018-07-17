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
from sklearn import cross_validation



####pyspark#####
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.util import MLUtils


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
    parser.add_argument("-f", "--folder", type=str, metavar="folder of features", help="folder contains features, hdfs://xxx.com:9000/user/fea", required=False)
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
        zip_file_name  = 'feat_impor_f.zip'

    # mongo info for connection
    mongo_tuples=(args.ip_address, args.port, args.db_name, args.tb_name, username, password)
    
    return feat_importance_firm(row_id_str, ds_id, hdfs_feat_dir, local_score_file
    , args.sp_master,config.get('spark', 'spark_rdd_compress'),config.get('spark', 'spark_driver_maxResultSize'), args.exe_memory, args.core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples
    , training_fraction, 'feature_importance_FRIM:'+row_id_str,  uploadtype)


# ================================================================================== train () ============ 
def feat_importance_firm(row_id_str, ds_id, hdfs_feat_dir, local_score_file
    , sp_master, spark_rdd_compress, spark_driver_maxResultSize, sp_exe_memory, sp_core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples
    , training_fraction, jobname, uploadtype): 

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
    
    
    '''
    SparkContext.setSystemProperty('spark.rdd.compress', config.get('spark', 'spark_rdd_compress'))
    SparkContext.setSystemProperty('spark.driver.maxResultSize', config.get('spark', 'spark_driver_maxResultSize'))
    SparkContext.setSystemProperty('spark.executor.memory', args.exe_memory)
    SparkContext.setSystemProperty('spark.cores.max', args.core_max)
    sc = SparkContext(args.sp_master, 'feature_importance_FRIM:'+str(args.row_id))
    '''    
    
    t0 = time()
    
    # get folder list (labels) from hdfs data_out/<id>/metadata  ==============
    dirFile_loc = os.path.join(hdfs_feat_dir , "metadata")
    dirFolders = sc.textFile(dirFile_loc)
    
    hash_Folders = dirFolders.collect()
    print "INFO: dirFile_loc=",dirFile_loc,", hash_Folders=",hash_Folders
    folder_list = [x.encode('UTF8') for x in hash_Folders]
    print "INFO: folder_list=",folder_list #['dirty/', 'clean/']
    

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

    #print "INFO: feature_count=",feature_count
    #print "dic_list=",dic_list #{u'123,345':u'136'}
    #print "dic_all_columns=",dic_all_columns # {1: u'8215,8216'}
    # end 
    
    # get {hash : raw string} mapping ==================================
    key = "dic_hash_str"  #{"123":"openFile"}
    jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
    jstr_proj='{"value":1}'
    # get parent dataset's data
    if ds_id != row_id_str:
        jstr_filter='{"rid":'+ds_id+',"key":"'+key+'"}'
        
    doc=query_mongo.find_one_t(mongo_tuples, jstr_filter, jstr_proj)
    dic_hash_str = doc['value']
    
    '''
    # get folder list (labels) from hdfs data_out/<id>/libsvm  ==============
    libsvm_loc = os.path.join(hdfs_feat_dir , "libsvm_data")    

    print "INFO: libsvm_loc=", libsvm_loc
    samples_rdd = MLUtils.loadLibSVMFile(sc, libsvm_loc)
    '''

    # filename for featured data
    libsvm_data_file = os.path.join(hdfs_feat_dir , "libsvm_data")
    print "INFO: libsvm_data_file=", libsvm_data_file
    
    # load feature count file
    #feat_count_file=libsvm_data_file+"_feat_count"
    #feature_count=zip_feature_util.get_feature_count(sc,feat_count_file)
    print "INFO: feature_count=",feature_count

    #samples_rdd = MLUtils.loadLibSVMFile(sc, libsvm_data_file)
    # load sample RDD from text file   
    #   also exclude selected features in sample ================ =====
    # format (LabeledPoint,hash) from str2LabeledPoint_hash() 
    samples_rdd, feature_count=zip_feature_util.get_sample_rdd(sc, libsvm_data_file, feature_count, excluded_feat_cslist=None)

    
    labels_and_features_rdd = samples_rdd.map(lambda p: (p[0].label, p[0].features))
    
    all_data = labels_and_features_rdd.collect()
    features_list = [x.toArray() for _,x in all_data]
    labels_list_all = [x for x,_ in all_data]
    labels_list_all = np.array(labels_list_all)
    features_array = np.array(features_list)

    ### generate sparse matrix (csr) for all samples
    features_sparse_mtx = csr_matrix(features_array)
    
    ### randomly split the samples into training and testing data
    sparse_mtx, sparse_test, labels_training, labels_testing = \
        cross_validation.train_test_split(features_sparse_mtx, labels_list_all, test_size=(1-training_fraction))
    
    #print "INFO: sparse_mtx.shape=",sparse_mtx.shape
    #print "INFO: sparse_test.shape=",sparse_test.shape
    row_num_training=(sparse_mtx.shape)[0]
    row_num_testing=(sparse_test.shape)[0]

    # why use LinearSVC ?
    clf = svm.LinearSVC()
    #clf = svm.SVC(C=0.1, kernel='rbf', degree=3, gamma=0.05, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
    #clf = svm.NuSVC(nu=0.3, kernel='rbf', degree=3, gamma=0.05, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False, max_iter=-1, random_state=None)
    #print "labels_training=",labels_training
    #print "sparse_mtx=",sparse_mtx
    clf.fit(sparse_mtx, labels_training)
    
    #print "**model:intercept***"
    #print clf.intercept_
    #print "**model:coef***"
    #print clf.coef_
    col_num=len(clf.coef_[0]) # for n_classes==2
    print "INFO: col_num=",col_num
    
    labels_pred = clf.predict(sparse_test)
    #print "labels_pred:", labels_pred
    
    accuracy = clf.score(sparse_test, labels_testing)
    print "INFO: data folder:", hdfs_feat_dir
    print "INFO: accuracy: ", accuracy
    
    #####################################################################
    ##################calculate feature importance with predication labels#######################
    #####################################################################
    AA = sparse_mtx.todense()
    BB = sparse_test.todense()
    labels_train_pred = clf.predict(sparse_mtx)
    labels_test_pred = labels_pred
    
    
    print "INFO: ###################################################################################"
    print "INFO: ############calculate feature importance with predication labels###################"
    print "INFO: ###################################################################################"
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
    
    print "INFO: =======Feature Importance(FIRM score)================"
    
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
            print "WARNING: No mapping found for feature number"
        
        str01 = str(feat)+"\t"+str(score)+"\t"+description_str+"\n"
        with open(local_score_file, "a") as f:
            f.write(str01)

    t1 = time()
    print 'INFO: running time: %f' %(t1-t0)
    
    
    
    
    print 'INFO: Finished: feature_importance_FRIM!'
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
