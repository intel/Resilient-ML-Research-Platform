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
from sklearn import svm, grid_search, datasets
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.svm import NuSVC
from sklearn.feature_selection import SelectKBest, chi2
#from sklearn.linear_model import LogisticRegression
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
from zip_feature_util import exclude_feature

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
    parser.add_argument("-o", "--out", type=str, metavar="out figure folder", help="folder contains output", required=False)
    parser.add_argument("-r", "--row_id", type=str, metavar="row id", help="row_id number in the db", required=False)
    parser.add_argument("-u", "--uploadtype", type=str, metavar="upload type", help="data type", required=False)
    parser.add_argument("-w", "--fromweb", type=str, metavar="flag for web", help="flag for web", required=False)
    parser.add_argument("-pm", "--parameter", type=str, metavar="parameters in json", help="json string contains learning alg and parameter selection", required=False)
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
    if args.row_id:
        row_id_str = args.row_id
    else:
        row_id_str  = '88'
    if args.out:
        local_out_dir = args.out
    else:
        local_out_dir  = 'out_result'
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
        ml_opts_jstr  = '{"learning_algorithm":"logistic_regression_with_lbfgs", "c":"1", "iteration":"300", "regularization":"l2"}'
    if len(args.username)>0:
        username = args.username
    else:
        username  = None
    if len(args.password)>0:
        password = args.password
    else:
        password  = None
    if args.bin:
        bin_number = eval(args.bin)
    else:
        bin_number  = 10
    if args.run:
        run_number = eval(args.run)
    else:
        run_number  = 2        
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
        zip_file_name  = 'train_MLlib_mrun.zip'

    # mongo info for connection
    mongo_tuples=(args.ip_address, args.port, args.db_name, args.tb_name, username, password)

    return mrun(row_id_str, ds_id, hdfs_feat_dir, local_out_dir, ml_opts_jstr, excluded_feat_cslist
    , args.sp_master,config.get('spark', 'spark_rdd_compress'),config.get('spark', 'spark_driver_maxResultSize'), args.exe_memory, args.core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples, fromweb
    , training_fraction, 'multi_run_mllib:'+row_id_str, run_number, bin_number )
    
    
# ================================================================================== train () ============ 
def mrun(row_id_str, ds_id, hdfs_feat_dir, local_out_dir, ml_opts_jstr, excluded_feat_cslist
    , sp_master, spark_rdd_compress, spark_driver_maxResultSize, sp_exe_memory, sp_core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples, fromweb
    , training_fraction, jobname, run_number, bin_number ): 
        
    ### generate data folder and out folder, clean up if needed
    if not os.path.exists(local_out_dir):
        os.makedirs(local_out_dir)

    # zip func in other files for Spark workers ================= ================
    zip_file_path = ml_build_zip_file(zipout_dir, zipcode_dir, zip_file_name, prefix='zip_feature_util')
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

    # check if ml_opts.has_excluded_feat ==1 ===================================
    has_excluded_feat=0
    if not ml_opts_jstr is None:
        ml_opts=json.loads(ml_opts_jstr)
        if "has_excluded_feat" in ml_opts:
            has_excluded_feat=ml_opts["has_excluded_feat"]
    #print "has_excluded_feat=",has_excluded_feat,",excluded_feat_cslist=",excluded_feat_cslist
    
    # get excluded feature list from mongo ========== ===
    if str(has_excluded_feat) == "1" and excluded_feat_cslist is None:
        key = "feature_excluded"
        jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
        jstr_proj='{"value":1}'
        # get from own id (not from parent dataset id)
        #print "jstr_filter=",jstr_filter,",jstr_proj=",jstr_proj
        doc=query_mongo.find_one(args.ip_address, args.port, args.db_name, args.tb_name, username, password, jstr_filter, jstr_proj)
        #print "feature_excluded=",doc
        if not doc is None and 'value' in doc:
            excluded_feat_cslist = ','.join(str(i) for i in doc['value'])
    print "INFO: excluded_feat_cslist=",excluded_feat_cslist
    
    
    ### generate Labeled point
    #libsvm_data_file = data_folder + "libsvm_data"
    # filename for featured data
    libsvm_data_file = os.path.join(hdfs_feat_dir , "libsvm_data")
    print "INFO: libsvm_data_file=", libsvm_data_file
    
    # load feature count file
    feat_count_file=libsvm_data_file+"_feat_count"
    feature_count=zip_feature_util.get_feature_count(sc,feat_count_file)
    print "INFO: feature_count=",feature_count

    #samples_rdd = MLUtils.loadLibSVMFile(sc, libsvm_data_file)
    # load sample RDD from text file   
    #   also exclude selected features in sample ================ =====
    # format (LabeledPoint,hash) from str2LabeledPoint_hash() 
    samples_rdd, feature_count=zip_feature_util.get_sample_rdd(sc, libsvm_data_file, feature_count, excluded_feat_cslist)
    #samples_rdd = MLUtils.loadLibSVMFile(sc, libsvm_data_file)
    
    # get distinct label list
    labels_list_all = samples_rdd.map(lambda p: p[0].label).distinct().collect()
    #labels_list_all = samples_rdd.map(lambda p: p.label).collect()

    t1 = time()
    print "INFO: labels_list_all=",labels_list_all
    #print "INFO: training and testing samples generated!"
    print 'INFO: data generating time: %f' %(t1-t0)
    t0 = t1
    
    ### generate label names (family names) #####
    ### connect to database to get the column list which contains all column number of the corresponding feature####
    label_set = set(labels_list_all)
    class_num = len(label_set)
    #class_num = len(labels_list)
    if class_num > 2:
        print "INFO:Number of classes=", class_num
    
    
    ###############################################
    ###########build learning model################
    ###############################################
    
    ### get the parameters###
    print "INFO: ============Learning Algorithm and Parameters============="
    #param_dict = json.loads(ml_opts_jstr)
    flag_model = ml_opts['learning_algorithm']     # 1: linear_svm_with_sgd; 2: logistic_regression_with_lbfgs; 3: logistic_regression_with_sgd
    C = eval(ml_opts['c'])
    iteration_num = ml_opts['iterations']
    regularization = ml_opts['regularization']
    print "INFO: Learning Algorithm: ", flag_model
    print "INFO: C = ", C
    print "INFO: iteration = ", iteration_num
    print "INFO: regType = ", regularization
    
    
    t0 = time()
    accuracy_array = np.zeros(run_number)
    for rnd in range (0, run_number):
    
        ### generate training and testing data
        training_rdd, testing_rdd = samples_rdd.randomSplit([training_fraction, 1-training_fraction])
        training_rdd=training_rdd.map(lambda p:p[0])# keep LabeledPoint only
        training_rdd.cache()
        testing_rdd.cache()
        training_sample_count = training_rdd.count()
                
        regP = C/float(training_sample_count)
        print "INFO: Calculated: regParam = ", regP
        
        ### build model ###
        
        if flag_model == "linear_svm_with_sgd":
            ### 1: linearSVM
            print "INFO: ====================1: Linear SVM============="
            model_classification = SVMWithSGD.train(training_rdd, regParam=regP, iterations=iteration_num, regType=regularization)   # regParam = 1/(sample_number*C)
            #print model_classification
        elif flag_model == "logistic_regression_with_lbfgs":
            ### 2: LogisticRegressionWithLBFGS
            print "INFO: ====================2: LogisticRegressionWithLBFGS============="
            model_classification = LogisticRegressionWithLBFGS.train(training_rdd, regParam=regP, iterations=iteration_num, regType=regularization, numClasses=class_num)   # regParam = 1/(sample_number*C)
        elif flag_model == "logistic_regression_with_sgd":
            ### 3: LogisticRegressionWithLBFGS
            print "INFO: ====================3: LogisticRegressionWithSGD============="
            model_classification = LogisticRegressionWithSGD.train(training_rdd, regParam=regP, iterations=iteration_num, regType=regularization)   # regParam = 1/(sample_number*C)    
        else:
            print "ERROR: Training model selection error: no valid ML model selected!"
            return
        
        ### Evaluating the model on testing data
        labelsAndPreds = testing_rdd.map(lambda p: (p[0].label, model_classification.predict(p[0].features)))
        labelsAndPreds.cache()
        testing_sample_number = testing_rdd.count()
        testErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(testing_sample_number)
        accuracy = 1 - testErr
        
        accuracy_array[rnd] = accuracy        
        print "INFO: current round=", rnd
        print "INFO: Accuracy=", accuracy
    

    ########################below: same as train_skLean_multi_run.py#####################################    
    ###############################################
    #######plot distribution and variance##########
    ###############################################

    plt.figure(1)
    
    num_bins = bin_number  ####10 is default
    n, bins, patches = plt.hist(accuracy_array, num_bins, normed=1, facecolor='green', alpha=0.5)
    ave = np.mean(accuracy_array)
    print "INFO: Accuracy mean=", ave
    variance = np.std(accuracy_array)
    print "INFO: Accuracy variance=", variance
    
    #print "INFO: bins: ", bins
    # add a 'best fit' line
    y = mlab.normpdf(bins, ave, variance)
    #print "INFO: y: ", y
    plt.plot(bins, y, 'r--')
    
    plt.title('Accuracy distribution of '+str(run_number)+' runs:')
    plt.xlabel('Accuracy Values')
    plt.ylabel('Probability / Accuracy bar width')
    
    #plt.savefig(local_out_dir+file_name_given+"_var_"+str(run_number)+".png")
    plt.savefig(os.path.join(local_out_dir, row_id_str+"_var_"+str(run_number)+".png"))

    # create ROC data for graph ====================
    all_json=[]
    barp_arr=[] #n
    disp_arr=[] #y
    last_idx=0
    for idx,ht in enumerate(n): # n is bar height
        #print "INFO: mrun bar=",idx, bins[idx], bins[idx+1],((bins[idx]+bins[idx+1])/2)
        barp_arr.append([ ((bins[idx]+bins[idx+1])/2.0),n[idx]]) # mid point for x axis
        if not math.isnan(y[idx]):
            disp_arr.append([bins[idx],y[idx]])
        last_idx=idx
    #print "INFO: ",last_idx+1, bins[last_idx+1], y[last_idx+1]
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
    
    mrun_jfile = os.path.join(local_out_dir, row_id_str+"_mrun.json")
    #print "INFO: all_json=",all_json
    print "INFO: mrun_jfile=",mrun_jfile
    if os.path.exists(mrun_jfile):
        try:
            os.remove(mrun_jfile)
        except OSError, e:
            print ("Error: %s - %s." % (e.mrun_jfile,e.strerror))

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
        print "INFO: mean="+str(mean*100)+"%"
        print "INFO: variance="+str(variance*100)+"%"

    
    print 'INFO: Finished!'
    return 0

    

    
if __name__ == '__main__':
    __description__ = "ML single run to show accuracy, roc, generate model, etc"
    main()
