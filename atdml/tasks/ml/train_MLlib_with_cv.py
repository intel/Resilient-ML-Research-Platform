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
import itertools

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

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.linalg import Vectors
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from operator import add


###pydoop###
import pydoop.hdfs as hdfs

#####matplotlib###############
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from ml_util import *
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
    parser.add_argument("-f", "--folder", type=str, metavar="folder of features", help="hdfs folder contains features", required=False)
    parser.add_argument("-n", "--name", type=str, metavar="file name", help="file name for sample folder", required=False)
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
        
    if args.fromweb:
        fromweb = args.fromweb
    else:
        fromweb  = None    
    if args.parameter:
        ml_opts_jstr = args.parameter
    else:
        ml_opts_jstr  = '{"learning_algorithm":"logistic_regression_with_lbfgs", "cv":"3", "mode":"expensive"}'
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
        zip_file_name  = 'train_MLlib_cv.zip'


    # mongo info for connection
    mongo_tuples=(args.ip_address, args.port, args.db_name, args.tb_name, username, password)

    return train(row_id_str, ds_id, hdfs_feat_dir, local_out_dir, ml_opts_jstr, excluded_feat_cslist
    , args.sp_master,config.get('spark', 'spark_rdd_compress'),config.get('spark', 'spark_driver_maxResultSize'), args.exe_memory, args.core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples, labelnameflag, fromweb
    , training_fraction, 'train_grid_search_mllib:'+row_id_str )
    
    
# ================================================================================== train () ============ 
def train(row_id_str, ds_id, hdfs_feat_dir, local_out_dir, ml_opts_jstr, excluded_feat_cslist
    , sp_master, spark_rdd_compress, spark_driver_maxResultSize, sp_exe_memory, sp_core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples, labelnameflag, fromweb
    , training_fraction, jobname ): 
    
        
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
    t00 = t0

    # check if ml_opts.has_excluded_feat ==1 ===================================
    has_excluded_feat=0
    ml_opts={}
    if not ml_opts_jstr is None:
        ml_opts=json.loads(ml_opts_jstr)
        if "has_excluded_feat" in ml_opts:
            has_excluded_feat=ml_opts["has_excluded_feat"]
    
    #print "has_excluded_feat=",has_excluded_feat,",excluded_feat_cslist=",excluded_feat_cslist
    
    # get excluded feature list from mongo ========== ===
    if str(has_excluded_feat) == "1" and excluded_feat_cslist is None:
        excluded_feat_cslist=ml_util.ml_get_excluded_feat(row_id_str, mongo_tuples)
    print "INFO: excluded_feat_cslist=",excluded_feat_cslist
    ### generate Labeled point
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
    samples_rdd,feature_count=zip_feature_util.get_sample_rdd(sc, libsvm_data_file, feature_count, excluded_feat_cslist)
    #samples_rdd = MLUtils.loadLibSVMFile(sc, libsvm_data_file)
    
    # get distinct label list
    labels_list_all = samples_rdd.map(lambda p: p[0].label).distinct().collect()
        
    ### generate training and testing data
    training_rdd, testing_rdd = samples_rdd.randomSplit([training_fraction, 1-training_fraction])
    training_rdd=training_rdd.map(lambda p:p[0])# keep LabeledPoint only
    training_rdd.cache() 
    training_sample_count = training_rdd.count()
    training_lbl_cnt_list=training_rdd.map(lambda p: (p.label,1)).reduceByKey(add).collect()
    testing_rdd.cache()
    testing_sample_count=testing_rdd.count()
    testing_lbl_cnt_list=testing_rdd.map(lambda p: (p[0].label,1)).reduceByKey(add).collect()
    sample_count=training_sample_count+testing_sample_count
            
    t1 = time()
    print "INFO: training sample count=",training_sample_count,", testing sample count=",testing_sample_count
    print "INFO: training label list=",training_lbl_cnt_list,", testing label list=",testing_lbl_cnt_list
    print "INFO: labels_list_all=",labels_list_all
    print "INFO: training and testing samples generated!"
    print 'INFO: running time: %f' %(t1-t0)
    t0 = t1
    
    
    ##############################################
    ########### Grid Search with CV ##############
    ##############################################
    
    ### get the parameters for cross validation and grid search ###
    (cv, model_name, param_dict) = generate_param(ml_opts)
    
    ### generate label names (family names) #####
    ### connect to database to get the column list which contains all column number of the corresponding feature####
    if labelnameflag == 1:
        label_dic=ml_util.ml_get_label_dict(row_id_str,mongo_tuples, ds_id)
        print "INFO: label_dic:", label_dic
    
    else:
        label_dic = {}
        label_set = set(labels_list_all)
        for label_value in label_set:
            label_dic[int(label_value)] = str(int(label_value))
        print "INFO: generated label_dic:", label_dic 
    
    labels_list = []
    for key in sorted(label_dic):
        labels_list.append(label_dic[key])
    #print "labels:", labels_list
    class_num = len(labels_list)
    if class_num > 2:
        print "INFO: Multi-class classification! Number of classes = ", class_num
    
    #### generate training and testing rdd(s) for CV#####
    split_prob = 1.0/float(cv)
    split_prob_list = []
    for i in range(0, cv):
        split_prob_list.append(split_prob)
    
    list_rdd = training_rdd.randomSplit(split_prob_list)
    list_train_rdd = []
    list_test_rdd = []
    for i in range(0, cv):
        list_rdd[i].cache()    
    for i in range(0, cv):
        tr_rdd = sc.emptyRDD()
        for j in range(0, cv):
            if j == i:
                pass
            else:
                tr_rdd = tr_rdd + list_rdd[j]
        tr_rdd.cache()
        list_train_rdd.append(tr_rdd)
        list_test_rdd.append(list_rdd[i])
    
    
    all_comb_list_of_dic = get_all_combination_list_of_dic(param_dict) 
    print "INFO: Total number of searching combinations:", len(all_comb_list_of_dic)
    
    ### loop for all parameter combinations and search the best parameters with CV###
    results = []
    for p in range(0, len(all_comb_list_of_dic)):
        params = all_comb_list_of_dic[p]
        C = params['C']
        iteration_num = params['iterations']
        regularization = params['regType']
        
        scores = []
        for i in range(0, cv):
            train_rdd = list_train_rdd[i]
            test_rdd = list_test_rdd[i]
            train_number = train_rdd.count()        
            regP = C/float(train_number)
            
            ### build model ###        
            if model_name == "linear_svm_with_sgd":
                #print "====================1: Linear SVM============="
                model_classification = SVMWithSGD.train(train_rdd, regParam=regP, iterations=iteration_num, regType=regularization)   # regParam = 1/(sample_number*C)
            elif model_name == "logistic_regression_with_lbfgs":
                #print "====================2: LogisticRegressionWithLBFGS============="
                model_classification = LogisticRegressionWithLBFGS.train(train_rdd, regParam=regP, iterations=iteration_num, regType=regularization, numClasses=class_num)   # regParam = 1/(sample_number*C)
            elif model_name == "logistic_regression_with_sgd":
                #print "====================3: LogisticRegressionWithSGD============="
                model_classification = LogisticRegressionWithSGD.train(train_rdd, regParam=regP, iterations=iteration_num, regType=regularization)   # regParam = 1/(sample_number*C)    
            else:
                print "ERROR: Training model selection error: no valid ML model selected!"
                return
   
            ### Evaluating the model on testing data
            labelsAndPreds = test_rdd.map(lambda p: (p.label, model_classification.predict(p.features)))
            labelsAndPreds.cache()
            test_sample_number = test_rdd.count()
            testErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(test_sample_number)
            accuracy = 1 - testErr
            #print "Accuracy = ", accuracy
            scores.append(accuracy)
        
        ss = np.asarray(scores)
        #print "%0.3f (+/-%0.03f) for " % (ss.mean(), ss.std() * 2), params
        results.append((ss.mean(), ss.std() * 2, params))
    
    sorted_results = sorted(results,key=lambda x: x[0], reverse=1)
    (best_accuracy, best_std2, best_param)  = sorted_results[0]
    print "INFO: ml_opts_jstr=",ml_opts_jstr
    print "INFO: best_param=",best_param
    
    #ml_opts=json.loads(ml_opts_jstr);
    print "INFO: ml_opts=",ml_opts
    
    ##############################################
    ######output Grid Search results##############
    ##############################################
    json2save={}
    json2save["rid"]=int(row_id_str)
    json2save["key"]="cv_result"
    #json2save["param_str"]=ml_opts_jstr
    json2save["param_dic"]=param_dict    
    cv_grid=[]
    print ""
    print "INFO: =====Grid Search Results for SPARK ======"
    print "INFO: Best parameters set found for ", model_name, " is: "
    for key in best_param:
        print "INFO:",key, "=", best_param[key]
        if key.lower()=="regtype":
            ml_opts['regularization']=str(best_param[key])
        else:
            ml_opts[key.lower()]=str(best_param[key]) # add best param to 
    ml_opts_jstr=json.dumps(ml_opts);
    json2save["param_str"]=ml_opts_jstr;
    print "INFO: Average accuracy with CV = ", cv, ": ", best_accuracy
    print ""
    print "INFO: Grid scores on development set:"
    for i in range(0, len(sorted_results)):
        (ave_accu_i, std2_i, param_i) = sorted_results[i]
        print "%0.3f (+/-%0.03f) for " % (ave_accu_i, std2_i), param_i
        #outstr='%s,%0.3f,%0.03f,%s' % (param_i,ave_accu_i, std2_i,"Selected" if param_i==best_param else "")
        outj={}
        outj["param"]=param_i
        outj["average_accuracy"]="%0.3f" % (ave_accu_i)
        outj["std_deviation"]="%0.3f" % (std2_i)
        outj["selected"]="%s" % ("Selected" if param_i==best_param else "")
        cv_grid.append(outj)
    print " "
    
    t1 = time()
    print 'INFO: Grid Search with CV run time: %f' %(t1-t0)
    t0 = time()
    
    ##################################################################################
    json2save["cv_grid_data"]=cv_grid
    cv_result=json.dumps(json2save)
    print "INFO: cv_result=",cv_result
    filter='{"rid":'+row_id_str+',"key":"cv_result"}'
    upsert_flag=True
    ## write to mongoDB.myml.dataset_info, ignore doc with duplicated key
    # db.dataset_info.createIndex({"rid":1,"key":1},{unique:true})
    ret=query_mongo.upsert_doc_t(mongo_tuples,filter,cv_result,upsert_flag)
    print "INFO: Upsert count for mllib cv_result: ret=",ret
    
    
    
    ############################################################################################
    ########### retrain with all training data and generate the final model with results #######
    ############################################################################################
    C = best_param['C']
    iteration_num = best_param['iterations']
    regularization = best_param['regType']
    regP = C/float(training_sample_count)
    
    ######################################the rest of the code is the same as train_MLlib.py #####################################################################
    
    if model_name == "linear_svm_with_sgd":
        ### 1: linearSVM
        print "INFO: ====================1: Linear SVM============="
        model_classification = SVMWithSGD.train(training_rdd, regParam=regP, iterations=iteration_num, regType=regularization)   # regParam = 1/(sample_number*C)
        #print model_classification
    elif model_name == "logistic_regression_with_lbfgs":
        ### 2: LogisticRegressionWithLBFGS
        print "INFO: ====================2: LogisticRegressionWithLBFGS============="
        model_classification = LogisticRegressionWithLBFGS.train(training_rdd, regParam=regP, iterations=iteration_num, regType=regularization, numClasses=class_num)   # regParam = 1/(sample_number*C)
    elif model_name == "logistic_regression_with_sgd":
        ### 3: LogisticRegressionWithSGD
        print "INFO: ====================3: LogisticRegressionWithSGD============="
        model_classification = LogisticRegressionWithSGD.train(training_rdd, regParam=regP, iterations=iteration_num, regType=regularization)   # regParam = 1/(sample_number*C)    
    else:
        print "INFO: Training model selection error: no valid ML model selected!"
        return


    print "INFO: model type=",type(model_classification)
    
    # create feature coefficient file ================================
    coef_arr=None
    intercept=None
    if model_classification.weights is None:
        print "WARNING: model weights not found!"
    else:
        coef_arr=model_classification.weights.toArray().tolist()
        # save to mongo
        key="coef_arr"
        ret=ml_util.save_json_t(row_id_str, key, coef_arr , mongo_tuples)
        # save intercept to mongo
        key="coef_intercept"
        intercept=model_classification.intercept 
        ret=ml_util.save_json_t(row_id_str, key, intercept, mongo_tuples)

        # feature list + coef file =============
        feat_filename=os.path.join(local_out_dir,row_id_str+"_feat_coef.json")
        print "INFO: feat_filename=",feat_filename
        
        # create feature list + coef file =============================================== ============
        # expect a dict of {"fid":(coef, feature_raw_string)}
        jret=ml_util.build_feat_list_t(row_id_str, feat_filename, None, None, coef_arr, ds_id  , mongo_tuples)
        
        # special featuring for IN or libsvm
        if jret is None:
            jret=ml_util.build_feat_coef_raw_list_t(row_id_str, feat_filename, coef_arr, ds_id, mongo_tuples)
        if jret is None:
            print "WARNING: Cannot create sample list for testing dataset. "
            
        jfeat_coef_dict=jret
        print "INFO: coef_arr len=",len(coef_arr), ", feature_count=",feature_count
        # for multi-class
        if len(coef_arr) != feature_count:
            jfeat_coef_dict={}
            print "WARNING: feature list can't be shown for multi-class classification"
        
        # Calculate prediction and Save testing dataset
        bt_coef_arr  = sc.broadcast( coef_arr)
        bt_intercept  = sc.broadcast(intercept)
        bt_jfeat_coef_dict  = sc.broadcast(jfeat_coef_dict)
        ### Evaluating the model on testing dataset: label, predict label, score, feature list
        print "INFO: intercept=",intercept
        print "INFO: coef_arr len=",len(coef_arr)
        print "INFO: jfeat_coef_dict len=",len(jfeat_coef_dict)


        # get prediction of testing dataset : (tlabel, plabel, score, libsvm, raw feat str, hash) ==============================
        if len(coef_arr) == feature_count:
            testing_pred_rdd = testing_rdd.map(lambda p: (
                 p[0].label \
                ,model_classification.predict(p[0].features) \
                ,zip_feature_util.calculate_hypothesis(p[0].features, bt_coef_arr.value, bt_intercept.value, model_name) \
                ,p[0].features \
                ,p[1] \
            ) ).cache()
        else: # for multi-class, no prediction score;, TBD for better solution: how to display multiple weights for each class
            testing_pred_rdd = testing_rdd.map(lambda p: (
                 p[0].label \
                ,model_classification.predict(p[0].features) \
                ,0 \
                ,p[0].features \
                ,p[1] \
            ) ).cache()

        # save false prediction to local file
        false_pred_fname=os.path.join(local_out_dir,row_id_str+"_false_pred.json")
        print "INFO: false_pred_fname=", false_pred_fname
        false_pred_data=testing_pred_rdd.filter(lambda p: p[0] != p[1])\
            .map(lambda p: (p[0],p[1],p[2] \
            ,zip_feature_util.get_dict_coef_raw4feat(zip_feature_util.sparseVector2dict(p[3]), bt_jfeat_coef_dict.value)
            ,p[4]  ) ) \
            .collect()
        print "INFO: false predicted count=", len(false_pred_data)
        false_pred_arr=[]
        with open (false_pred_fname,"w")as fp:
            for sp in false_pred_data:
                jsp={"tlabel":sp[0],"plabel":sp[1],"score":sp[2],"feat":sp[3],"hash":sp[4]}
                #print "jsp=",jsp
                false_pred_arr.append(jsp)
            fp.write(json.dumps(false_pred_arr))
            
        # save prediction results, format: label, prediction, hash
        pred_ofname=os.path.join(local_out_dir,row_id_str+"_pred_output.pkl")
        print "INFO: pred_ofname=", pred_ofname
        pred_out_arr=testing_pred_rdd.map(lambda p: (p[0],p[1],p[4])).collect()
        ml_util.ml_pickle_save(pred_out_arr,pred_ofname)

        
    ### Evaluating the model on testing data
    #labelsAndPreds = testing_rdd.map(lambda p: (p.label, model_classification.predict(p.features)))
    labelsAndPreds = testing_pred_rdd.map(lambda p: (p[0],p[1]) )
    labelsAndPreds.cache()
    #testing_sample_count = testing_rdd.count()
    testErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(testing_sample_count)
    accuracy = 1 - testErr
    print "INFO: Accuracy = ", accuracy
    
    ### Save model
    #save_dir = config.get('app', 'HADOOP_MASTER')+'/user/hadoop/yigai/row_6/'
    #save_dir = config.get('app', 'HADOOP_MASTER')+config.get('app', 'HDFS_MODEL_DIR')+'/'+row_id_str
    save_dir = os.path.join(config.get('app', 'HADOOP_MASTER'),config.get('app', 'HDFS_MODEL_DIR'),row_id_str)
    try:
        hdfs.ls(save_dir)
        #print "find hdfs folder"
        hdfs.rmr(save_dir)
        #print "all files removed"
    except IOError as e:
        print "WARNING: I/O error({0}): {1}".format(e.errno, e.strerror),". At HDFS=", save_dir
    except:
        print "WARNING: Unexpected error:", sys.exc_info()[0] ,". At HDFS=", save_dir
    
    model_classification.save(sc, save_dir)
        
    ###load model if needed 
    #sameModel = SVMModel.load(sc, save_dir)
    
    t1 = time()
    print 'INFO: training run time: %f' %(t1-t0)
    t0 = t1
    
    ###############################################
    ###########plot prediction result figure#######
    ###############################################
    
    
    labels = labelsAndPreds.collect()
    true_label_list = [x for x,_ in labels]
    pred_label_list = [x for _,x in labels]

    pred_fname=os.path.join(local_out_dir,row_id_str+"_1"+".png")
    true_fname=os.path.join(local_out_dir,row_id_str+"_2"+".png")
    pred_xlabel='Prediction (Single Run)'
    true_xlabel='True Labels (Single Run)'
    test_cnt_dic=ml_util.ml_plot_predict_figures(pred_label_list, true_label_list, labels_list, label_dic, testing_sample_count 
        , pred_xlabel, pred_fname, true_xlabel, true_fname)
    
    
    plt.show()
    perf_measures=None
    dataset_info={"training_fraction":training_fraction, "class_count":class_num,"dataset_count":sample_count}
    #############################################################
    ###################for 2 class only (plot ROC curve)#########
    #############################################################
    if len(labels_list) == 2:


        do_ROC=True
        reverse_label_dic = dict((v,k) for k, v in label_dic.items())
        if 'clean' in reverse_label_dic:
            flag_clean = reverse_label_dic['clean']
        elif 'benign' in reverse_label_dic:
            flag_clean = reverse_label_dic['benign']
        elif '0' in reverse_label_dic:
            flag_clean = 0
        else:
            print "WARNING: No ROC curve generated: 'clean' or '0' must be a label for indicating negative class!"
            do_ROC=False

        # build data file for score graph
        score_graph_fname=os.path.join(local_out_dir,row_id_str+"_score_graph.json")
        print "INFO: score_graph_fname=", score_graph_fname
        
        # build score_arr_0, score_arr_1
        #    format: tlabel, plabel, score, libsvm, raw feat str, hash
        graph_arr=testing_pred_rdd.map(lambda p: (int(p[0]), float(p[2])) ).collect()
        score_arr_0=[]    
        score_arr_1=[]    
        max_score=0
        min_score=0
        for p in graph_arr:
            if p[0]==0:
                score_arr_0.append(p[1])
            else:
                score_arr_1.append(p[1])
            # save max,min score
            if p[1]>max_score:
                max_score=p[1]
            elif p[1]<min_score:
                min_score=p[1]
        
        ml_build_pred_score_graph(score_arr_0, score_arr_1, model_name, score_graph_fname, max_score,min_score)
        #print "score_arr_0=",score_arr_0
        #print "score_arr_1=",score_arr_1
        #print "max_score=",max_score
        #print "min_score=",min_score
            
        if do_ROC:
        
            perf_measures=ml_util.calculate_fscore(true_label_list, pred_label_list)
            print "RESULT: perf_measures=",perf_measures
            model_classification.clearThreshold()
            scoreAndLabels = testing_rdd.map(lambda p: (model_classification.predict(p[0].features), int(p[0].label)))
            #metrics = BinaryClassificationMetrics(scoreAndLabels)
            #areROC = metrics.areaUnderROC
            #print areROC
            scoreAndLabels_list = scoreAndLabels.collect()
            if flag_clean == 0:
                scores = [x for x,_ in scoreAndLabels_list]
                s_labels = [x for _,x in scoreAndLabels_list]
                testing_N = test_cnt_dic[0]
                testing_P = test_cnt_dic[1]
            else:
                scores = [-x for x,_ in scoreAndLabels_list]
                s_labels = [1-x for _,x in scoreAndLabels_list]
                testing_N = test_cnt_dic[1]
                testing_P = test_cnt_dic[0]
            #print scores
            #print s_labels
            # create ROC data file ======== ==== 
            roc_auc=ml_create_roc_files(row_id_str, scores,s_labels,testing_N,testing_P
                , local_out_dir, row_id_str)
                
            perf_measures["roc_auc"]=roc_auc
    
    # only update db for web request
    if fromweb=="1": 
        #print "database update"
        str_sql="UPDATE atdml_document set "+"accuracy = '"+str(accuracy*100)+"%" \
            +"', status = 'learned', processed_date ='"+str(datetime.datetime.now()) \
            +"',ml_opts='"+ml_opts_jstr \
            +"', perf_measures='"+json.dumps(perf_measures) \
            +"', dataset_info='"+json.dumps(dataset_info) \
            +"' where id="+row_id_str 
        ret=exec_sqlite.exec_sql(str_sql)
        print "INFO: Data update done! ret=", str(ret)
    else:
        print "INFO: accuracy = '"+str(accuracy*100)+"%"
    
    
    t1 = time()
    print 'INFO: total run time: %f' %(t1-t00)
    
    print 'INFO: Finished!'
    return 0

def generate_param(param_dict):
    
    #param_dict = json.loads(ml_opts_jstr)
    model_name = param_dict['learning_algorithm']     # 1: linear_svm; 2: ; 3: 
    cv = eval(param_dict['cv'])
    mode = param_dict['mode']
    
    print ""
    print "INFO: ============Learning Algorithm and Grid Search Parameters============="    
    print "INFO: Learning Algorithm: ", model_name
    print "INFO: CV = ", cv
    print "INFO: mode = ", mode
    
    ###parse and print print parameters###
        
    
    if model_name == "linear_svm_with_sgd":
        ### 1: Linear SVM
        if mode == "cheap":
            param_dict = [{'C': [0.0001, 0.01, 1, 100, 10000], 'regType':['l2'], 'iterations':[300]}]
        else:
            param_dict = [{'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'regType':['l2', 'l1'], 'iterations':[500]}]
        print "INFO: Grid Search Parameters:"
        print "INFO: C = ", param_dict[0]['C']
        print "INFO: regType = ", param_dict[0]['regType']                        
    elif model_name == "logistic_regression_with_lbfgs" or model_name == "logistic_regression_with_sgd" :
        ### 2: LogisticRegressionWithLBFGS; 3: LogisticRegressionWithSGD
        if mode == "cheap":
            param_dict = [{'C': [0.0001, 0.01, 1, 100, 10000], 'regType':['l2'], 'iterations':[300]}]
        else:
            param_dict = [{'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'regType':['l2', 'l1'], 'iterations':[500]}]         
        print "INFO: Grid Search Parameters:"
        print "INFO: C = ", param_dict[0]['C']
        print "INFO: regType = ", param_dict[0]['regType'] 
    else:
        print "INFO: Training model selection error: no valid ML model selected!"
        return (0, "none", 0, 0, 0)
    return (cv, model_name, param_dict)  

def get_all_combination_list_of_dic(param_dict):
    all_comb_list_of_dic = []
    for p in range (0, len(param_dict)):
        dic = param_dict[p]
        key_list = []
        list_of_list = []
        for key in dic:
            key_list.append(key)
            list_of_list.append(dic[key])
        all_combinations = list(itertools.product(*list_of_list))
        
        for i in range(0, len(all_combinations)):
            curr_dic = {}
            for j in range(0, len(key_list)):
                key = key_list[j]
                val = all_combinations[i][j]
                curr_dic[key] = val
            all_comb_list_of_dic.append(curr_dic)

    return all_comb_list_of_dic

    
if __name__ == '__main__':
    __description__ = "ML training by cross validation"
    main()
