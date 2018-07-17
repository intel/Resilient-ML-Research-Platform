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
from operator import add

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
        zip_file_name  = 'train_MLlib.zip'
 
    # mongo info for connection
    mongo_tuples=(args.ip_address, args.port, args.db_name, args.tb_name, username, password)
    
    return train(row_id_str, ds_id, hdfs_feat_dir, local_out_dir, ml_opts_jstr, excluded_feat_cslist
    , args.sp_master,config.get('spark', 'spark_rdd_compress'),config.get('spark', 'spark_driver_maxResultSize'), args.exe_memory, args.core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples, labelnameflag, fromweb
    , training_fraction, 'train_mllib:'+row_id_str, random_seed=config.get('machine_learning', 'random_seed') )
    
    
# ================================================================================== train () ============ 
def train(row_id_str, ds_id, hdfs_feat_dir, local_out_dir, ml_opts_jstr, excluded_feat_cslist
    , sp_master, spark_rdd_compress, spark_driver_maxResultSize, sp_exe_memory, sp_core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples, labelnameflag, fromweb
    , training_fraction, jobname, random_seed=None ): 
 
    ### generate data folder and out folder, clean up if needed
    #local_out_dir = local_out_dir + "/"
    #if os.path.exists(local_out_dir): 
    #    shutil.rmtree(local_out_dir) # to keep smaplelist file
    if not os.path.exists(local_out_dir):
        os.makedirs(local_out_dir)
            
    # create zip files for Spark workers ================= ================
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
       
    # filename for featured data
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
    samples_rdd, feature_count=zip_feature_util.get_sample_rdd(sc, libsvm_data_file, feature_count, excluded_feat_cslist)

    # get distinct label list
    labels_list_all = samples_rdd.map(lambda p: p[0].label).distinct().collect()
    
    # split samples to training and testing data, format (LabeledPoint,hash)
    training_rdd, testing_rdd = samples_rdd.randomSplit([training_fraction, 1-training_fraction ], seed=int(random_seed))
    training_rdd=training_rdd.map(lambda p:p[0])# keep LabeledPoint only
    training_rdd.cache() 
    training_sample_count = training_rdd.count()
    training_lbl_cnt_list=training_rdd.map(lambda p: (p.label,1)).reduceByKey(add).collect()
    testing_rdd.cache()
    testing_sample_count=testing_rdd.count()
    testing_lbl_cnt_list=testing_rdd.map(lambda p: (p[0].label,1)).reduceByKey(add).collect()
    sample_count=training_sample_count+testing_sample_count
    
    # random_seed testing 
    if not random_seed is None:
        all_t=testing_rdd.collect()
        all_t=sorted(all_t,key=lambda x:x[1])
        cnt=0
        for i in all_t:
            print i[1]
            cnt=cnt+1
            if cnt >3:
                break
    
    t1 = time()
    print "INFO: training sample count=",training_sample_count,", testing sample count=",testing_sample_count
    print "INFO: training label list=",training_lbl_cnt_list,", testing label list=",testing_lbl_cnt_list
    print "INFO: labels_list_all=",labels_list_all
    print "INFO: training and testing samples generated!"
    print 'INFO: running time: %f' %(t1-t0)
    t0 = t1
    
    ###############################################
    ###########build learning model################
    ###############################################
    
    ### get the parameters###
    print "INFO: ======Learning Algorithm and Parameters============="
    #ml_opts = json.loads(ml_opts_jstr)
    model_name = ml_opts['learning_algorithm']     # 1: linear_svm_with_sgd; 2: logistic_regression_with_lbfgs; 3: logistic_regression_with_sgd
    iteration_num=0
    if 'iterations' in ml_opts:
        iteration_num = ml_opts['iterations']
    C=0
    if 'c' in ml_opts:
        C = eval(ml_opts['c'])
    regularization = ""
    if 'regularization' in ml_opts:
        regularization = ml_opts['regularization']
    
    print "INFO: Learning Algorithm: ", model_name
    print "INFO: C = ", C
    print "INFO: iterations = ", iteration_num
    print "INFO: regType = ", regularization
    regP = C/float(training_sample_count)
    print "INFO: Calculated: regParam = ", regP
    
    ### generate label names (family names) #####
    ### connect to database to get the column list which contains all column number of the corresponding feature####
    if labelnameflag == 1:
        '''
        key = "dic_name_label"
        jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
        jstr_proj='{"value":1}'
 
        # get parent dataset's data
        if ds_id != row_id_str:
            jstr_filter='{"rid":'+ds_id+',"key":"'+key+'"}'
 
        doc=query_mongo.find_one_t(mongo_tuples, jstr_filter, jstr_proj)
        dic_list = doc['value']
        print "INFO: dic_list=",dic_list
        
        label_dic = {}
        for i in range(0, len(dic_list)):
            for key in dic_list[i]:
                label_dic[dic_list[i][key]] = key.encode('UTF8')
        '''
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
    print "INFO: labels:", labels_list
    class_num = len(labels_list)
    if class_num > 2:
        print "INFO: Multi-class classification! Number of classes = ", class_num

    
    ### build model ###
    
    if model_name == "linear_svm_with_sgd":
        ### 1: linearSVM
        print "INFO: ====================1: Linear SVM============="
        model_classification = SVMWithSGD.train(training_rdd, regParam=regP
            , iterations=iteration_num, regType=regularization)   # regParam = 1/(sample_number*C)
        #print model_classification
    elif model_name == "logistic_regression_with_lbfgs":
        ### 2: LogisticRegressionWithLBFGS
        print "INFO: ====================2: LogisticRegressionWithLBFGS============="
        model_classification = LogisticRegressionWithLBFGS.train(training_rdd, regParam=regP
            , iterations=iteration_num, regType=regularization, numClasses=class_num)   # regParam = 1/(sample_number*C)
    elif model_name == "logistic_regression_with_sgd":
        ### 3: LogisticRegressionWithSGD
        print "INFO: ====================3: LogisticRegressionWithSGD============="
        model_classification = LogisticRegressionWithSGD.train(training_rdd, regParam=regP
            , iterations=iteration_num, regType=regularization)   # regParam = 1/(sample_number*C)    
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
        coef_weights=model_classification.weights
        #print "coef_weights=",coef_weights
        #print type(coef_weights),coef_weights.shape
        coef_arr=coef_weights.toArray().tolist()
        # save coef_arr to mongo
        key="coef_arr"
        ret=ml_util.save_json_t(row_id_str, key, coef_arr , mongo_tuples)
        
        # save coef_arr to local file
        if ret==0:
            # drop old record in mongo
            filter='{"rid":'+row_id_str+',"key":"coef_arr"}'
            ret=query_mongo.delete_many(mongo_tuples,None,filter)
            if not os.path.exists(local_out_dir):
                os.makedirs(local_out_dir)
            fn_ca=os.path.join(local_out_dir,row_id_str,row_id_str+"_coef_arr.pkl")
            print 
            ml_util.ml_pickle_save(coef_arr, fn_ca)
            
            
        
        # save intercept to mongo
        intercept=model_classification.intercept
        key="coef_intercept"
        ret=ml_util.save_json_t(row_id_str, key, intercept, mongo_tuples)
            
        # feature list + coef file =============
        feat_filename=os.path.join(local_out_dir,row_id_str+"_feat_coef.json")
        print "INFO: feat_filename=",feat_filename
        
        # create feature, coef & raw string file =============================================== ============
        # expect a dict of {"fid":(coef, feature_raw_string)}
        jret=ml_util.build_feat_list_t(row_id_str, feat_filename, None, None, coef_arr, ds_id, mongo_tuples)
        
        # special featuring for IN or libsvm
        if jret is None:
            jret=ml_util.build_feat_coef_raw_list_t(row_id_str, feat_filename, coef_arr, ds_id, mongo_tuples)
        if jret is None:
            print "WARNING: Cannot create sample list for testing dataset. "
        
        jfeat_coef_dict=jret
        print "INFO: coef_arr len=",len(coef_arr), ", feature_count=",feature_count
        # for multi-class
        if len(coef_arr) != feature_count :
            jfeat_coef_dict={}
            print "WARNING: coef count didn't match feature count.  multi-class classification was not supported"


        # Calculate prediction and Save testing dataset
        bt_coef_arr  = sc.broadcast( coef_arr)
        bt_intercept  = sc.broadcast(intercept)
        bt_jfeat_coef_dict  = sc.broadcast(jfeat_coef_dict)
        ### Evaluating the model on testing dataset: label, predict label, score, feature list
        print "INFO: intercept=",intercept
        print "INFO: coef_arr len=",len(coef_arr), type(coef_arr)
        print "INFO: jfeat_coef_dict len=",len(jfeat_coef_dict) #, jfeat_coef_dict
        
        # get prediction of testing dataset : (tlabel, plabel, score, libsvm, raw feat str, hash) ==============================
        if len(coef_arr) == feature_count:
            testing_pred_rdd = testing_rdd.map(lambda p: (
                 p[0].label \
                ,model_classification.predict(p[0].features) \
                ,zip_feature_util.calculate_hypothesis(p[0].features, bt_coef_arr.value, bt_intercept.value, model_name) \
                ,p[0].features \
                ,p[1] \
            ) ).cache()
        else: # for multi-class, no prediction score; TBD for better solution: how to display multiple weights for each class
            testing_pred_rdd = testing_rdd.map(lambda p: (
                 p[0].label \
                ,model_classification.predict(p[0].features) \
                ,"-" \
                ,p[0].features \
                ,p[1] \
            ) ).cache()
        ''',p[0].features.dot(bt_coef_arr.value)+bt_intercept.value \
        # Save testing dataset for analysis
        libsvm_testing_output = hdfs_feat_dir + "libsvm_testing_output_"+row_id_str
        print "INFO: libsvm_testing_output=", libsvm_testing_output
        try:
            hdfs.rmr(libsvm_testing_output)
        except IOError as e:
            print "WARNING: I/O error({0}): {1}".format(e.errno, e.strerror)
        except:
            print "WARNING: Unexpected error at libsvm_testing_output file clean up:", sys.exc_info()[0] 
        # save only false prediction?
        #testing_pred_rdd.filter(lambda p: p[0] != p[1]).saveAsTextFile(libsvm_testing_output)
        testing_pred_rdd.saveAsTextFile(libsvm_testing_output)
        
        '''
        #test_tmp=testing_pred_rdd.collect()
        
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
        '''
        one_item= testing_pred_rdd.first()
        print "one_item=",one_item
        sparse_arr=one_item[3]

        dict_feat=zip_feature_util.sparseVector2dict(sparse_arr)
        print "len=",len(dict_feat),"dict_feat=",dict_feat
        dict_weit=zip_feature_util.add_coef2dict(coef_arr,dict_feat)
        print "len=",len(dict_weit),"dict_weit=",dict_weit
        '''
    # Calculate Accuracy. labelsAndPreds = (true_label,predict_label)
    labelsAndPreds = testing_pred_rdd.map(lambda p: (p[0],p[1]) )
    labelsAndPreds.cache()
    testing_sample_number = testing_rdd.count()
    testErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(testing_sample_number)
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
    ###########plot prediction result figure ==================================================== ===============
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
    print "INFO: figure files: ", pred_fname, true_fname
    #print "INFO: Number of samples in each label is=", test_cnt_dic

    roc_auc=None
    perf_measures=None
    dataset_info={"training_fraction":training_fraction, "class_count":class_num,"dataset_count":sample_count}
    #############################################################
    ###################for 2 class only (plot ROC curve) ==================================================== ===============
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
            print "INFO: No ROC curve generated: 'clean','benign' or '0' must be a label for indicating negative class!"
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

            
        if do_ROC:
        
            perf_measures=ml_util.calculate_fscore(true_label_list, pred_label_list)
            print "RESULT: perf_measures=",perf_measures

            '''
            # calculate fscore  ==========
            tp = labelsAndPreds.filter(lambda (v, p): v == 1 and p==1 ).count() 
            fp = labelsAndPreds.filter(lambda (v, p): v == 0 and p==1 ).count() 
            fn = labelsAndPreds.filter(lambda (v, p): v == 1 and p==0 ).count() 
            tn = labelsAndPreds.filter(lambda (v, p): v == 0 and p==0 ).count() 
            print "RESULT: tp=",tp,",fp=",fp,",fn=",fn,",tn=",tn
            precision=float(tp)/(tp+fp)
            recall=float(tp)/(tp+fn)
            print "RESULT: precision=",precision,",recall=",recall
            acc=(tp+tn)/(float(testing_sample_number))
            fscore=2*((precision*recall)/(precision+recall))
            print "RESULT: fscore=",fscore,",acc=",acc  
            '''
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
            
            # create ROC data file ======== ==== 
            roc_auc=ml_create_roc_files(row_id_str, scores,s_labels,testing_N,testing_P
            , local_out_dir, row_id_str)
            #, local_out_dir, file_name_given)
            
            perf_measures["roc_auc"]=roc_auc

    # only update db for web request ==================================================== ===============
    if fromweb=="1": 
        #print "database update"
        str_sql="UPDATE atdml_document set "+"accuracy = '"+str(accuracy*100)+"%" \
            +"', status = 'learned', processed_date ='"+str(datetime.datetime.now()) \
            +"', perf_measures='"+json.dumps(perf_measures) \
            +"', dataset_info='"+json.dumps(dataset_info) \
            +"' where id="+row_id_str
        ret=exec_sqlite.exec_sql(str_sql)
        print "INFO: Data update done! ret=", str(ret)
    else:
        print "INFO: accuracy = '"+str(accuracy*100)+"%"

    
    print 'INFO: Finished!'
    return 0

    
if __name__ == '__main__':
    __description__ = "ML single run to show accuracy, roc, generate model, etc"
    main()
