#! /usr/bin/python
'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''
#
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
import json, pickle, gzip

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
from sklearn.decomposition import RandomizedPCA
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc

####pyspark#####
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionWithSGD
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.tree import DecisionTree

###pydoop###
import pydoop.hdfs as hdfs

#####matplotlib###############
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

#####import for django database####
sys.path.append('./db')
import exec_sqlite
import query_mongo
import ml_util
from ml_util import *
import zip_feature_util 

CONF_FILE='../../app.config' # at the base dir of the web
config=ConfigParser.ConfigParser()
config.read(CONF_FILE)
libsvm_filename = config.get("machine_learning","libsvm_alldata_filename")
dnn_filename = config.get("machine_learning","dnn_alldata_filename")



def main():
    
    parser = ArgumentParser(description=__description__)
    parser.add_argument("-f", "--folder", type=str, metavar="folder of features", help="hdfs folder contains features", required=False)
    parser.add_argument("-n", "--data_fname", type=str, metavar="data file name", help="file name for sample dataset", required=False)
    parser.add_argument("-o", "--out", type=str, metavar="out figure folder", help="folder contains output", required=False)
    parser.add_argument("-r", "--row_id", type=str, metavar="row id", help="row_id number in the db", required=False)
    parser.add_argument("-w", "--fromweb", type=str, metavar="flag for web", help="flag for web", required=False)
    parser.add_argument("-pm", "--parameter", type=str, metavar="parameters in json", help="json string contains learning alg and parameter selection", required=False)
    parser.add_argument("-sl", "--showlabelname", type=str, metavar="show label name", help="0: not shown; 1: show label name", required=False)
    parser.add_argument("-dsid", "--ds_id", type=str, metavar="source dataset id", help="source dataset id for training option", required=False)

    
    parser.add_argument('-sp','--sp_master', type=str, dest='sp_master', help='spark.master'
                , default =config.get('spark', 'spark_master'))
    parser.add_argument('-em','--exe_memory', type=str, dest='exe_memory', help='spark.executor.memory'
                , default =config.get('spark', 'spark_executor_memory'))
    parser.add_argument('-cm','--core_max', type=str, dest='core_max', help='spark.cores.max'
                , default =config.get('spark', 'spark_cores_max'))
    
    #### get label/family name from mongo
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
                
    # zip spark code for feature exclusion
    parser.add_argument("-ef", "--excluded_feat_cslist", type=str, metavar="feature ids to be excluded", help="feature id list to be excluded", required=False)
    parser.add_argument("-zo", "--zipout_dir", type=str, metavar="out figure folder", help="folder contains python code", required=False)
    parser.add_argument("-zc", "--zipcode_dir", type=str, metavar="out code", help="out code folder for python code", required=False)
    parser.add_argument("-zf", "--zipfilename", type=str, metavar="python code zip file", help="python code zip file for distribution to works", required=False)
    parser.add_argument("-tf", "--training_fraction", type=str, metavar="fraction for training set"
                , dest='training_fraction', help="fraction for training set", required=False)
    
    args = parser.parse_args()
    
    if args.training_fraction:
        training_fraction = args.training_fraction
    else:
        training_fraction  = eval(config.get("machine_learning","training_portion"))
    if args.folder:
        hdfs_feat_dir = args.folder
    else:
        hdfs_feat_dir  = '.'
    if args.row_id:
        row_id_str = args.row_id
    else:
        row_id_str  = '88'
    if args.out:
        local_out_dir = args.out
    else:
        local_out_dir  = '.'
    if args.ds_id:
        ds_id = args.ds_id
    else:
        ds_id  = ''

    if args.fromweb:
        fromweb = args.fromweb
    else:
        fromweb  = None    
    if args.parameter:
        ml_opts_jstr = args.parameter
    else:
        ml_opts_jstr  = '{"learning_algorithm":"logistic_regression_with_lbfgs", "c":"1", "iterations":"300", "regularization":"l2"}'
    if args.showlabelname: # mllib or scikit
        labelnameflag = eval(args.showlabelname)
    else:
        labelnameflag = 0    
    if len(args.username)>0:
        username = args.username
    else:
        username  = None
    if len(args.password)>0:
        password = args.password
    else:
        password  = None 
    # exclude feature
    if args.excluded_feat_cslist:
        excluded_feat_cslist = args.excluded_feat_cslist
    else:
        excluded_feat_cslist  = None 
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
        zip_file_name  = 'preprocess_dnn.zip'
    if args.data_fname:
        data_fname = args.data_fname
    else:
        data_fname  = libsvm_filename
 
    # mongo info for connection
    #mongo_tuples=(args.ip_address, args.port, args.db_name, args.tb_name, username, password)
    
    return preprocess(row_id_str, ds_id, hdfs_feat_dir, local_out_dir, ml_opts_jstr #, excluded_feat_cslist
    , args.sp_master,config.get('spark', 'spark_rdd_compress'),config.get('spark', 'spark_driver_maxResultSize'), args.exe_memory, args.core_max
    , zipout_dir, zipcode_dir, zip_file_name, data_fname
    #, mongo_tuples, labelnameflag, fromweb
    , 'prepross DNN:'+row_id_str )
    
    
# ================================================================================== train () ============ 
def preprocess(row_id_str, ds_id, hdfs_feat_dir, local_out_dir, ml_opts_jstr #, excluded_feat_cslist
    , sp_master, spark_rdd_compress, spark_driver_maxResultSize, sp_exe_memory, sp_core_max
    , zipout_dir, zipcode_dir, zip_file_name, data_fname
    #, mongo_tuples, labelnameflag, fromweb
    , jobname 
    , dnn_data_suffix=config.get("machine_learning","dnn_data_suffix")
    , dnn_label_suffix=config.get("machine_learning","dnn_label_suffix")
    , dnn_info_suffix=config.get("machine_learning","dnn_info_suffix")
	# add param for chunk size, multivariant etc.
	#     , cust_featuring=None, cust_featuring_params=None
	
    ): 
 
    ### generate data folder and out folder, clean up if needed

    if not os.path.exists(local_out_dir):
        os.makedirs(local_out_dir)
            
    # create zip files for Spark workers ================= ================
    zip_file_path = ml_util.ml_build_zip_file(zipout_dir, zipcode_dir, zip_file_name, prefix='zip_feature_util')
    print "INFO: zip_file_path=",zip_file_path, ",ml_opts_jstr=", ml_opts_jstr

    # get_spark_context
    sc=ml_util.ml_get_spark_context(sp_master
        , spark_rdd_compress
        , spark_driver_maxResultSize
        , sp_exe_memory
        , sp_core_max
        , jobname
        , [zip_file_path]) 

    t0 = time()
    


    # Check option
    ml_opts={}
    try:
        ml_opts=json.loads(ml_opts_jstr)
    except:
        print "ERROR: string ml_opts is invalid!"
        return -1

    learning_algorithm=ml_opts["learning_algorithm"]
    #return    

        

    # create 3 arrays
    feat_list=[]
    label_list=[]
    info_list=[]
    feature_count=None
    sample_count=None
    if learning_algorithm =="cnn": #==================================================  ==========
        # libsvm_data for featured data, not use "dnn_data" here
        libsvm_data_file = os.path.join(hdfs_feat_dir , "libsvm_data")
        print "INFO: libsvm_data_file=", libsvm_data_file
        # load sample RDD from libsvm file   
        # output format: [([features], label, info)] , feature_count, feat_max, feat_min
        all_list, feature_count, feat_max, feat_min =\
            zip_feature_util.get_sample_as_arr(sc, libsvm_data_file, None, None)

        c=0
        # convert to 3 list  =========================  ==========
        for i in all_list:
            #print type(i[0]),type(i[1]), type(i[2])
            if c % 10000 ==0:
                print ".",
                sys.stdout.flush()
            c=c+1
            # convert to float32 np array and normalize by max value (0-1.0) 
            feat_list.append(np.array(i[0],dtype=np.float32)/feat_max)
            label_list.append(i[1])
            info_list.append(i[2])
        print "INFO: feat_list t=",type(feat_list), type(feat_list[0]), type(feat_list[0][0])
    elif learning_algorithm =="lstm": #==================================================  ==========
        # filename for featured data
        dnn_data_file = os.path.join(hdfs_feat_dir , data_fname)
        print "INFO: dnn_data_file=", dnn_data_file
        # output format: [([features], label, info)] , chucked_sample_count
        all_list, chucked_sample_count=\
            zip_feature_util.get_sample_as_chunk(sc, dnn_data_file, None, None)
        c=0
        # convert to 3 list  =========================  ==========
        for i in all_list:
            #print type(i[0]),type(i[1]), type(i[2])
            if c % 10000 ==0:
                print ".",
                sys.stdout.flush()
            c=c+1
            # convert to float32 np array and normalize by max value (0-1.0) 
            feat_list.append(np.array(i[0],dtype=np.int32))
            label_list.append(i[1])
            info_list.append(i[2])
        print "INFO: feat_list t=",type(feat_list), type(feat_list[0]), feat_list[0][0]
    print "INFO: label_list t=",type(label_list), label_list[0]
    print "INFO: info_list t=",type(info_list),info_list[0]            
    
    if not all_list is None and len(all_list)>0:
        sample_count=len(all_list)
    else:
        sample_count=None
    print "INFO: sample_count=",sample_count

    tgt_prefix=row_id_str
    if (row_id_str != ds_id):
        tgt_prefix=ds_id

    # save data file   =========================  ==========     
    data_fname=os.path.join(local_out_dir,tgt_prefix+dnn_data_suffix)
    print "INFO: data_fname=", data_fname
    # reshape for image
    #if not feature_count is None and feature_count >0 and not sample_count is None and sample_count > 0:
    #    nparr_feat=np.asarray(feat_list,dtype=np.float32).reshape(sample_count,feature_count)
    #else: # lstm didn't need to reshape
    nparr_feat=np.asarray(feat_list,dtype=np.int32) 
    print "INFO: nparr_feat s=",nparr_feat.shape , nparr_feat[0].shape, type(nparr_feat[0][0]),nparr_feat[0][0]

    with gzip.open(data_fname,"wb")as fp:
        #pickle.dump(nparr_feat,fp, 2) # may overflow in pickle; limited to 2B elements; use numpy.save()
        np.save(fp,nparr_feat,allow_pickle=False)    #62.9M for (465724, 2967)


    # save label file  =========================  ==========
    lbl_fname=os.path.join(local_out_dir,tgt_prefix+dnn_label_suffix)
    print "INFO: lbl_fname=", lbl_fname
    # convert to numpy array and int32 to save space
    nparr_lbl=np.asarray(label_list,dtype=np.int32)
    #print "label=",nparr_lbl[0]
    with gzip.open(lbl_fname,"wb")as fp:
        #pickle.dump(nparr_lbl,fp,  2)
        np.save(fp,nparr_lbl,allow_pickle=False)             
    # save info file  =========================  ==========  
    info_fname=os.path.join(local_out_dir,tgt_prefix+dnn_info_suffix)
    print "INFO: info_fname=", info_fname
    nparr_info=np.asarray(info_list)
    with gzip.open(info_fname,"wb")as fp:
        #pickle.dump(nparr_info,fp,2)
        np.save(fp,nparr_info,allow_pickle=False)             
            
    

    
    t1 = time()
    print 'INFO: running time: %f' %(t1-t0)
    print 'INFO: Finished!'
    return 0

    
if __name__ == '__main__':
    __description__ = "ML single run to show accuracy, roc, generate model, etc"
    main()
