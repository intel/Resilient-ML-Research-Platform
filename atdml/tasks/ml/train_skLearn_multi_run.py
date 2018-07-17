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
import json

from argparse import ArgumentParser
from scipy.sparse import *
from scipy import *
from time import time
from sklearn import svm, linear_model, naive_bayes, grid_search, datasets
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
from sklearn.datasets import load_svmlight_file
from sklearn import cross_validation

####pyspark#####
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint

#####matplotlib###############
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

#####import for django database####
sys.path.append('./db')
import exec_sqlite

####import our own library####
import query_mongo
import ml_util
from ml_util import *
import zip_feature_util
from zip_feature_util import exclude_feature

CONF_FILE='../../app.config' # at the base dir of the web
config=ConfigParser.ConfigParser()
config.read(CONF_FILE)
training_portion = eval(config.get("machine_learning","training_portion"))
mtx_name_list = config.get("machine_learning","mtx_name_list")
mtx_libsvm = config.get("machine_learning","mtx_libsvm")
mtx_stat = config.get("machine_learning","mtx_stat")


def main():
    
    parser = ArgumentParser(description=__description__)
    parser.add_argument("-f", "--folder", type=str, metavar="folder of features", help="folder contains features, hdfs://xxx.com:9000/user/fea", required=False)
    parser.add_argument("-n", "--name", type=str, metavar="file name", help="file name for sample folder", required=False)
    parser.add_argument("-o", "--out", type=str, metavar="out figure folder", help="folder contains output", required=False)
    parser.add_argument("-r", "--row_id", type=str, metavar="row id", help="row_id number in the db", required=False)
    parser.add_argument("-u", "--uploadtype", type=str, metavar="upload type", help="data type", required=False)
    parser.add_argument("-w", "--fromweb", type=str, metavar="flag for web", help="flag for web", required=False)
    parser.add_argument("-pm", "--parameter", type=str, metavar="parameters in json", help="json string contains learning alg and parameter selection", required=False)
    #parser.add_argument("-sl", "--showlabelname", type=str, metavar="show label name", help="0: not shown; 1: show label name", required=False)
    parser.add_argument("-dsid", "--ds_id", type=str, metavar="source dataset id", help="source dataset id for training option", required=False)

    parser.add_argument("-b", "--bin", type=str, metavar="bin number", help="number of bins for var plot", required=False)
    parser.add_argument("-mn", "--run", type=str, metavar="run number", help="number of runs for var plot", required=False)
    
    # Spark
    parser.add_argument('-sp','--sp_master', type=str, dest='sp_master', help='spark.master'
                , default =config.get('spark', 'spark_master'))
    parser.add_argument('-em','--exe_memory', type=str, dest='exe_memory', help='spark.executor.memory'
                , default =config.get('spark', 'spark_executor_memory'))
    parser.add_argument('-cm','--core_max', type=str, dest='core_max', help='spark.cores.max'
                , default =config.get('spark', 'spark_cores_max'))

    #### get dic_name_label from mongo db
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
        training_fraction  = training_portion
    if args.folder:
        hdfs_feat_dir = args.folder
    else:
        hdfs_feat_dir  = config.get('app', 'HADOOP_MASTER')+'/user/hadoop/yigai/sality_virut_zbot_backdoor_hash_000'
    if args.name:
        file_name_given = args.name
    else:
        file_name_given  = 'aaaa'
    if args.out:
        local_out_dir = args.out
    else:
        local_out_dir  = 'out_result'
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
    if args.fromweb:
        fromweb = args.fromweb
    else:
        fromweb  = None
    if args.parameter:
        ml_opts_jstr = args.parameter
    else:
        ml_opts_jstr  = '{"learning_algorithm":"linear_svm", "c":"1", "regularization":"l2", "kernel":"rbf", "gamma":"0", "degree":"3", "nu":"0.1"}'
    if args.bin:
        bin_number = eval(args.bin)
    else:
        bin_number  = 10
    if args.run:
        run_number = eval(args.run)
    else:
        run_number  = 2    


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
        zip_file_name  = 'train_sklean_mrun.zip'

    # mongo info for connection
    mongo_tuples=(args.ip_address, args.port, args.db_name, args.tb_name, username, password)

    return mrun(row_id_str, ds_id, hdfs_feat_dir, local_out_dir, ml_opts_jstr, excluded_feat_cslist
    , args.sp_master,config.get('spark', 'spark_rdd_compress'),config.get('spark', 'spark_driver_maxResultSize'), args.exe_memory, args.core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples, fromweb
    , training_fraction, 'multi_run_scikit:'+row_id_str, run_number, bin_number )


# ================================================================================== train () ============ 
def mrun(row_id_str, ds_id, hdfs_feat_dir, local_out_dir, ml_opts_jstr, excluded_feat_cslist
    , sp_master, spark_rdd_compress, spark_driver_maxResultSize, sp_exe_memory, sp_core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples, fromweb
    , training_fraction, jobname, run_number, bin_number): 

    
    if not os.path.exists(local_out_dir):
        os.makedirs(local_out_dir)
    
    # zip func in other files for Spark workers ================= ================
    zip_file_path = ml_build_zip_file(zipout_dir, zipcode_dir, zip_file_name, prefix='zip_feature_util')
    print "INFO: zip_file_path=",zip_file_path

    # init Spark context ====
    sc=ml_util.ml_get_spark_context(sp_master
        , spark_rdd_compress
        , spark_driver_maxResultSize
        , sp_exe_memory
        , sp_core_max
        , jobname
        , [zip_file_path]) 
    
    t0 = time()
    
    # check if ml_opts.has_excluded_feat ==1 ===================================
    has_excluded_feat=0
    if not ml_opts_jstr is None:
        ml_opts=json.loads(ml_opts_jstr)
        if "has_excluded_feat" in ml_opts:
            has_excluded_feat=ml_opts["has_excluded_feat"]
            
    # get excluded feature list from mongo ========== ===
    if str(has_excluded_feat) == "1" and excluded_feat_cslist is None:
        excluded_feat_cslist=ml_util.ml_get_excluded_feat(row_id_str, mongo_tuples)
    print "INFO: excluded_feat_cslist=",excluded_feat_cslist
    
    ### load libsvm file ###
    #libsvm_data_file = data_folder + "libsvm_data"
    # source libsvm filename  
    libsvm_data_file = os.path.join(hdfs_feat_dir , "libsvm_data")
    print "INFO: libsvm_data_file:", libsvm_data_file

    # load feature count file
    feat_count_file=libsvm_data_file+"_feat_count"
    feature_count=zip_feature_util.get_feature_count(sc,feat_count_file)
    print "INFO: feature_count=",feature_count
    
    # load sample RDD from text file   
    #   also exclude selected features in sample ================ =====
    # format (LabeledPoint,hash) from str2LabeledPoint_hash() 
    #samples_rdd = MLUtils.loadLibSVMFile(sc, libsvm_data_file)
    samples_rdd, feature_count = zip_feature_util.get_sample_rdd(sc, libsvm_data_file, feature_count, excluded_feat_cslist)
    #samples_rdd = MLUtils.loadLibSVMFile(sc, libsvm_data_file)
    #print samples_rdd.count()
    

    # collect all data to local for processing ===============
    all_data = samples_rdd.map(lambda p:p[0]).collect()  # keep LabeledPoint only
    # 2-D array
    features_list = [x.features.toArray() for x in all_data]
    # label array
    labels_list_all = [x.label for x in all_data]

    # convert to np array
    labels_list_all = array(labels_list_all)
    features_array = np.array(features_list)
    
    ### generate sparse matrix (csr) for all samples
    features_sparse_mtx = csr_matrix(features_array)
    
    t1 = time()
    print "INFO: labels_list_all=",labels_list_all
    print 'INFO: data generating time: %f' %(t1-t0)
    
    label_set = set(labels_list_all)
    class_num = len(label_set)
    #class_num = len(labels_list)
    if class_num > 2:
        print "INFO: Number of classes =", class_num
    
    
    ###############################################
    ###########build learning model################
    ###############################################
    
    ### parse parameters and generate the model ###
    (clf, model_name) = parse_para_and_get_model(ml_opts)
    if model_name == "none":
        return
    
    t0 = time()
    accuracy_array = np.zeros(run_number)
    for rnd in range (0, run_number):
        
        ### randomly split the samples into training and testing data
        X_train_sparse, X_test_sparse, labels_train, labels_test = \
            cross_validation.train_test_split(features_sparse_mtx, labels_list_all, test_size=(1-training_fraction))
           
        #####fit the model to training dataset ####
        try:
            clf.fit(X_train_sparse, labels_train)
        except ValueError as v:
            print "INFO: ValueError:", v
            #raise v didn't work
            return -1
            
        ### Evaluating the model on testing data
        labels_pred = clf.predict(X_test_sparse)
                
        accuracy = clf.score(X_test_sparse, labels_test)        
        accuracy_array[rnd] = accuracy
        
        print "INFO: current round: ", rnd
        print "INFO: Accuracy = ", accuracy
    
    
    ###############################################
    #######plot distribution and variance##########
    ###############################################

    plt.figure(1)
    
    num_bins = bin_number  ####10 is default
    n, bins, patches = plt.hist(accuracy_array, num_bins, normed=1, facecolor='green', alpha=0.5)
    ave = np.mean(accuracy_array)
    print "INFO: num_bins=",num_bins
    print "INFO: accuracy_array=",accuracy_array
    print "INFO: Accuracy mean: ", ave
    variance = np.std(accuracy_array)
    print "INFO: Accuracy variance: ", variance
    
    # add a 'best fit' line
    y = mlab.normpdf(bins, ave, variance)
    #print "INFO: y: ", y
    plt.plot(bins, y, 'r--')
    
    plt.title('Accuracy distribution of '+str(run_number)+' runs:')
    plt.xlabel('Accuracy Values')
    plt.ylabel('Probability / Accuracy bar width')
    mrun_fname=os.path.join(local_out_dir, row_id_str+"_var_"+str(run_number)+".png")
    plt.savefig(mrun_fname)
    
    # create data for graph ====================
    all_json=[]
    barp_arr=[] #n
    disp_arr=[] #y
    last_idx=0
    for idx,ht in enumerate(n): # n is bar height
        #print "mrun bar=",idx, bins[idx], bins[idx+1],((bins[idx]+bins[idx+1])/2)
        barp_arr.append([ ((bins[idx]+bins[idx+1])/2.0),n[idx]]) # mid point for x axis, may shift 100% bar
        if not math.isnan(y[idx]):
            disp_arr.append([bins[idx],y[idx]])
        last_idx=idx
    #print last_idx+1, bins[last_idx+1], y[last_idx+1]
    if not math.isnan(y[last_idx+1]):
        disp_arr.append([bins[last_idx+1],y[last_idx+1]])
    #print "barp_arr=", barp_arr
    #print "disp_arr=", disp_arr
    #bar
    bar_json={}
    bar_json["values"]=barp_arr
    bar_json["key"]='Mutil-Run Accuracy' #
    bar_json["type"]="bar" # light blue
    bar_json["yAxis"]=1
    all_json.append(bar_json)
    #distribution
    if len(disp_arr)>0:
        dis_json={}
        dis_json["values"]=disp_arr
        dis_json["key"]='Normal Distribution' #
        dis_json["type"]="line" # light blue
        dis_json["yAxis"]=1
        all_json.append(dis_json)
    
    mrun_jfile=os.path.join(local_out_dir, row_id_str+"_mrun.json")
    #mrun_jfile = local_out_dir+row_id_str+"_mrun.json"
    #print "INFO: all_json=",all_json
    print "INFO: mrun_jfile=",mrun_jfile
    if os.path.exists(mrun_jfile):
        try:
            os.remove(mrun_jfile)
        except OSError, e:
            print ("ERROR: %s - %s." % (e.mrun_jfile,e.strerror))

    try:
        with open(mrun_jfile,"w") as json_file:
            json.dump(all_json, json_file)
    except Exception as e:
        print "ERROR: ",e

    
    
    t1 = time()
    print 'INFO: running time: %f' %(t1-t0)
    
    # only update db for web request
    if fromweb=="1": 
        #print "database update"
        str_sql="UPDATE atdml_document set "+"mean = '"+str(ave*100)+"%"+"', variance = '"+str(variance*100) \
            +"%',status = 'mruned', processed_date ='"+str(datetime.datetime.now()) \
            +"' where id="+row_id_str
        ret=exec_sqlite.exec_sql(str_sql)
        print "INFO: Data update done! ret=", str(ret)
    else:
        print "INFO: mean = '"+str(mean*100)+"%"
        print "INFO: variance = '"+str(variance*100)+"%"
        
    

    

def parse_para_and_get_model(param_dict):
    
    #param_dict = json.loads(ml_opts_jstr)
    model_name = param_dict['learning_algorithm']     # 1: linear_svm; 2: ; 3: 
    
    ###parse and print print parameters###
    print "INFO: ============Learning Algorithm and Parameters============="    
    print "INFO: param_dict=",param_dict
    if model_name == "linear_svm":
        ### 1: linearSVM
        C = eval(param_dict['c'])
        C = float(C)
        print "INFO: Learning Algorithm: ", model_name
        print "INFO: C = ", C        
        print "INFO: ====================1: Linear SVM============="
        clf = svm.LinearSVC(C=C)
    
    elif model_name == "svm":
        ### 2: SVM with kernel        
        C = eval(param_dict['c'])
        C = float(C)
        kernel_func = param_dict['kernel']
        gamma_val = "0.0"
        if 'gamma' in param_dict:
            gamma_val=eval(param_dict['gamma'])
            gamma_val = float(gamma_val)
        print "INFO: Learning Algorithm: ", model_name
        print "INFO: C = ", C
        print "INFO: kernel = ", kernel_func
        print "INFO: gamma = ", gamma_val
        if kernel_func == "poly":
            degree_num = eval(param_dict['degree'])
            print "degree = ", degree_num        
        print "INFO: ====================2: SVM with kernel============="
        if kernel_func == "poly":
            clf = svm.SVC(C=C, kernel=kernel_func, gamma = gamma_val, degree = degree_num)
        elif kernel_func == "rbf" or kernel_func == "sigmoid":
            clf = svm.SVC(C=C, kernel=kernel_func, gamma = gamma_val)
        else:
            clf = svm.SVC(C=C, kernel=kernel_func)
    
    elif model_name == "nu_svm":
        ### 3: NuSVC
        nu_val = eval(param_dict['nu'])
        nu_val = float(nu_val)
        kernel_func = param_dict['kernel']
        gamma_val = eval(param_dict['gamma'])
        gamma_val = float(gamma_val)
        print "INFO: Learning Algorithm: ", model_name
        print "INFO: nu = ", nu_val
        print "INFO: kernel = ", kernel_func
        print "INFO: gamma = ", gamma_val
        if kernel_func == "poly":
            degree_num = eval(param_dict['degree'])
            print "INFO: degree = ", degree_num                
        print "INFO: ====================3: NuSVC============="
        if kernel_func == "poly":
            clf = svm.NuSVC(nu=nu_val, kernel=kernel_func, gamma = gamma_val, degree = degree_num)
        elif kernel_func == "rbf" or kernel_func == "sigmoid":
            clf = svm.NuSVC(nu=nu_val, kernel=kernel_func, gamma = gamma_val)
        else:
            clf = svm.NuSVC(nu=nu_val, kernel=kernel_func)
    
    elif model_name == "logistic_regression":
        ### 4: linearSVM
        C = eval(param_dict['c'])
        C = float(C)
        # penalty from CV, regularization from non-CV training
        if 'regularization' in param_dict:
            regularization = param_dict['regularization']
        elif 'penalty' in param_dict:
            regularization = param_dict['penalty']
        print "INFO: Learning Algorithm: ", model_name
        print "INFO: C = ", C
        print "INFO: penalty = ", regularization                
        print "INFO: ====================4: Logistic Regression============="
        clf = linear_model.LogisticRegression(C=C, penalty=regularization)
        
    elif model_name == "linear_svm_with_sgd":
        ### 5: linearSVM with SGD, no para as input
        print "INFO: Learning Algorithm: ", model_name                
        print "INFO: ====================5: Linear SVM with SGD============="
        clf = linear_model.SGDClassifier()
    elif model_name == "passive_aggressive_classifier":
        ### 6: Passive Aggressive Classifier
        C = eval(param_dict['c'])
        C = float(C)
        print "INFO: Learning Algorithm: ", model_name
        print "INFO: C = ", C        
        print "INFO: ====================6: Passive Aggressive Classifier============="
        clf = linear_model.PassiveAggressiveClassifier(C=C)
    elif model_name == "perceptron":
        ### 7: Perceptron
        print "INFO: Learning Algorithm: ", model_name                
        print "INFO: ====================7: Perceptron============="
        clf = linear_model.Perceptron()
    else:
        print "INFO: Training model selection error: no valid ML model selected!"
        return (0, "none")
    return (clf, model_name)
    
    
if __name__ == '__main__':
    __description__ = "ML single run to show accuracy, roc, generate model, etc"
    main()
