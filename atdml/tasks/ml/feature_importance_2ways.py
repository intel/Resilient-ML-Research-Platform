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


####pyspark#####
from pyspark import SparkContext
from pyspark.sql import SQLContext

#####matplotlib###############
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

####global constant

####import our own library####
sys.path.append('./db')
import query_mongo
import ml_util

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
    parser.add_argument("-c", "--column", type=str, metavar="column number", help="column number in the table", required=False)
    parser.add_argument("-f", "--folder", type=str, metavar="folder of features", help="hdfs folder contains features", required=False)
    parser.add_argument("-n", "--name", type=str, metavar="file name", help="file name for sample folder", required=False)
    parser.add_argument("-r", "--row_id", type=str, metavar="row_id number", help="row_id number in the db", required=False)
    
    parser.add_argument("-u", "--uploadtype", type=str, metavar="upload type", help="data type", required=False)
    parser.add_argument("-df", "--desc_file", type=str, metavar="feature description mapping file", help="feature description mapping file", required=False)
    parser.add_argument("-dsid", "--ds_id", type=str, metavar="source dataset id", help="source dataset id for training option", required=False)

    parser.add_argument("-pb", "--scoreprob", type=str, metavar="output score file (probability)", help="file name for output score (probability)", required=False)
    parser.add_argument("-it", "--scoreIT", type=str, metavar="output score file (Information Gain)", help="file name for output score (Information Gain)", required=False)

    parser.add_argument('-sp','--sp_master', type=str, dest='sp_master', help='spark.master'
                , default =config.get('spark', 'spark_master'))
    parser.add_argument('-em','--exe_memory', type=str, dest='exe_memory', help='spark.executor.memory'
                , default =config.get('spark', 'spark_executor_memory'))
    parser.add_argument('-cm','--core_max', type=str, dest='core_max', help='spark.cores.max'
                , default =config.get('spark', 'spark_cores_max'))
    
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
    if args.scoreprob:
        score_file_prob = args.scoreprob
    else:
        score_file_prob  = 'score_Prob.txt'
    if args.scoreIT:
        score_file_IT = args.scoreIT
    else:
        score_file_IT  = 'score_IT.txt'
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
        zip_file_name  = 'feat_impor2w.zip'

    # mongo info for connection
    mongo_tuples=(args.ip_address, args.port, args.db_name, args.tb_name, username, password)
    
    return feat_importance_2way(row_id_str, ds_id, hdfs_feat_dir, score_file_IT, score_file_prob
    , args.sp_master,config.get('spark', 'spark_rdd_compress'),config.get('spark', 'spark_driver_maxResultSize'), args.exe_memory, args.core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples
    , 'feature_importance_2ways:'+row_id_str,  uploadtype, description_file)


# ================================================================================== train () ============ 
def feat_importance_2way(row_id_str, ds_id, hdfs_feat_dir, score_file_IT, score_file_prob
    , sp_master, spark_rdd_compress, spark_driver_maxResultSize, sp_exe_memory, sp_core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples
    , jobname, uploadtype, description_file): 

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
    
    # get folder list (labels) from hdfs data_out/<id>/metadata
    dirFile_loc = os.path.join(hdfs_feat_dir , "metadata")
    dirFolders = sc.textFile(dirFile_loc)
    
    hash_Folders = dirFolders.collect()
    #print "hash_Folders=",hash_Folders
    folder_list = [x.encode('UTF8') for x in hash_Folders]
    #print "INFO: dirFile_loc=",dirFile_loc,", folder_list=",folder_list
        
    row_num_training = 0

    
    sample_numbers = []
    sparse_mtx_list = []
    features_list = []
    row_list = []
    col_list = []
    max_feat_list = []
    
    for folder in folder_list:
        print "INFO: folder:", folder
        label = folder_list.index(folder) + 1
        print 'INFO: label=', label
        # md5 list
        logFile_name = os.path.join(hdfs_feat_dir,folder, mtx_name_list)
        # libsvm data
        logFile_data = os.path.join(hdfs_feat_dir, folder, mtx_libsvm)
        #logFile_data = hdfs_feat_dir + folder + mtx_stat
        #print "INFO: logFile_name=",logFile_name
        #print "INFO: logFile_data=",logFile_data

        logNames = sc.textFile(logFile_name).cache()
        logData = sc.textFile(logFile_data).cache()
        
        names = logNames.collect()
        data = logData.collect()
        name_l = [x.encode('UTF8') for x in names]
        feature_l = [x.encode('UTF8') for x in data]
        name_list = [names.strip() for names in name_l]
        feature_list = [features.strip() for features in feature_l]

        num_names = len(name_list)
        sample_numbers.append(num_names)
        print 'INFO: sample_numbers=', sample_numbers
        print 'INFO: name_list count=', num_names

        
        ########generate list for csc_matrix creation #########
        i = 0;
        
        features_training = []
        row_training = []
        col_training = []
        labels_training = []
        max_feat= 0
        while i < num_names:

            features = feature_list[i]
            
            features = features.strip()
            feature_array = features.split(' ')
            labels_training.append(label)
            
            length = len(feature_array)
            j = 0
            while j < length:
                feature = feature_array[j]
                feat, value = feature.split(':', 2)
                row_training.append(i)
                col_training.append(int(feat) - 1)
                features_training.append(int(value))
                max_feat = max(max_feat, int(feat))
                j = j+1
            i = i+1
        
        
        #print "features_training=",features_training
        #print "row_training=",row_training
        #print "col_training=",col_training
        features_training = array(features_training)
        row_training = array(row_training)
        col_training = array(col_training)
        
        #print "type(row_training)=",type(row_training),",row_training=",row_training
        #print "type(col_training)=",type(col_training),",col_training=",col_training
        
        max_feat_list.append(max_feat)
        features_list.append(features_training)
        row_list.append(row_training)
        col_list.append(col_training)

        row_num_training = row_num_training + num_names
    #print " END for folder in folder_list: "  
        
    col_num = max(max_feat_list)
    print "INFO: column number=", col_num, ", len(max_feat_list)=",len(max_feat_list)
    
    for i in range(0, len(max_feat_list)):
        sparse_mtx = csc_matrix((features_list[i],(row_list[i],col_list[i])), shape=(sample_numbers[i],col_num))
        sparse_mtx_list.append(sparse_mtx)
    
    #print sparse_mtx_list[0]
    #print "sparse_mtx_list[0].shape=",sparse_mtx_list[0].shape
    #print sparse_mtx_list[1]
    #print "sparse_mtx_list[1].shape=",sparse_mtx_list[1].shape
    
    
    exclusive_feature_set_mal = []
    exclusive_feature_set_clean = []
    dic_feature_cnt_mal = {}
    dic_feature_cnt_clean = {}
    
    dic_score = {}
    dic_cnt_mal = {}
    dic_cnt_clean = {}
    
    dic_IT_grain = {}

    
        
    ####################################################
    ####feature importance algorithms: 2 methods ####### # Only for 2 classes ???
    ####################################################
    if len(sample_numbers) == 2:
           
        ###################################################
        ################## calculate probability ############
        ###################################################
        
        
        print "INFO: ======= Feature Importance(probability) ================"
                
        for j in range(0, col_num):

            curr_col_dirty = sparse_mtx_list[0].getcol(j)
            sum_col = curr_col_dirty.sum(0)
            cnt_mal = sum_col.tolist()[0][0]
            
            curr_col_clean = sparse_mtx_list[1].getcol(j)
            sum_col = curr_col_clean.sum(0)
            cnt_clean = sum_col.tolist()[0][0]
            
            percnt_mal = cnt_mal/float(sample_numbers[0])
            percnt_clean = cnt_clean/float(sample_numbers[1])
            score_j = (percnt_mal + 1 - percnt_clean)/2
            
            dic_score[j+1] = score_j
            dic_cnt_clean[j+1] = cnt_clean
            dic_cnt_mal[j+1] = cnt_mal
        
        sorted_score = sorted(dic_score.items(), key=operator.itemgetter(1), reverse=True)
        
        #print "sorted_score:", sorted_score
        #print "dic_cnt_clean", dic_cnt_clean
        #print "dic_cnt_mal", dic_cnt_mal
    

        
        if os.path.exists(score_file_prob):
            try:
                os.remove(score_file_prob)
            except OSError, e:
                print ("Error: %s - %s." % (e.score_file_prob,e.strerror))
        
        for ii in range (0, len(sorted_score)):
            (feat, score) = sorted_score[ii]
            #print feat, score, dic_all_columns[feat]
                        
            str01 = str(feat)+"\t"+str(score)+"\t"+description_str+"\n"
            with open(score_file_prob, "a") as f:
                f.write(str01)

        ########################################################
        ##################Information Gain (entropy)############
        ########################################################
       
        print "INFO: ======= Information Gain ================"
        for j in range(0, col_num):
            cnt_mal = dic_cnt_mal[j+1]               
            cnt_clean = dic_cnt_clean[j+1]
                        
            
            total_samples = sample_numbers[0] + sample_numbers[1]
            
            p0 = float(sample_numbers[0])/total_samples
            p1 = 1-p0
            
            if p0 == 0 or p1 == 0:
                parent_entropy = 0
            else:
                parent_entropy = 0 - p0*np.log2(p0) - p1*np.log2(p1)
        
            if cnt_clean+cnt_mal == 0:
                information_gain = 0
            elif total_samples - cnt_clean - cnt_mal == 0:
                information_gain = 0
            else:
                p0 = float(cnt_mal)/(cnt_clean+cnt_mal)
                p1 = 1-p0
                if p0 == 0 or p1 == 0:
                    child_left_entropy = 0
                else: 
                    child_left_entropy = 0 - p0*np.log2(p0) - p1*np.log2(p1)
            
                p0 = float(sample_numbers[0]- cnt_mal)/(total_samples - cnt_clean - cnt_mal)
                p1 = 1-p0
                if p0 == 0 or p1 == 0:
                    child_right_entropy = 0
                else:
                    child_right_entropy = 0 - p0*np.log2(p0) - p1*np.log2(p1)
                
                weighted_child_entropy = child_left_entropy * float(cnt_clean+cnt_mal)/total_samples + child_right_entropy * float(total_samples - cnt_clean - cnt_mal)/total_samples
                information_gain = parent_entropy - weighted_child_entropy
            
            dic_IT_grain[j+1] = information_gain

        sorted_IT_gain = sorted(dic_IT_grain.items(), key=operator.itemgetter(1), reverse=True)

        
        if os.path.exists(score_file_IT):
            try:
                os.remove(score_file_IT)
            except OSError, e:
                print ("ERROR: %s - %s." % (e.score_file_IT,e.strerror))
        
        for ii in range (0, len(sorted_IT_gain)):
            (feat, score) = sorted_IT_gain[ii]
                        
            str01 = str(feat)+"\t"+str(score)+"\t"+description_str+"\n"
            with open(score_file_IT, "a") as f:
                f.write(str01)

                
    t1 = time()
    print 'INFO: running time: %f' %(t1-t0)
        
    
    print 'INFO: Finished!'
    return 0

   
if __name__ == '__main__':
    __description__ = "feature importance: Information gain and probability"
    main()
