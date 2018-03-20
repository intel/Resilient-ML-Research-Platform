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
import gzip
import json, traceback

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
from sklearn.svm import NuSVC, classes, LinearSVC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import RandomizedPCA
from sklearn.externals import joblib

#####import for django database####
sys.path.append('./db')
import query_mongo
import exec_sqlite
import ml_util

CONF_FILE='../../app.config' # at the base dir of the web
config=ConfigParser.ConfigParser()
config.read(CONF_FILE)

def main():

    parser = ArgumentParser(description=__description__)
    parser.add_argument("-d", "--name", type=str, metavar="file name", help="file name for prediction", required=False)
    parser.add_argument("-o", "--out", type=str, metavar="learner output", help="out files for prediction", required=False)
    parser.add_argument("-r", "--row_id", type=str, metavar="row_id number", help="row_id number in the db", required=False)
    parser.add_argument("-i", "--cid", type=str, metavar="child row id", help="child row id for prediction", required=False)
    ####other parameters
    parser.add_argument("-nb", "--num", type=str, metavar="n gram", help="window size for n gram", required=False)
    parser.add_argument("-pa", "--para", type=str, metavar="param in 1 gram", help="number of parameters in 1 gram, if -1, no 1 gram", required=False)
    parser.add_argument("-x", "--max", type=str, metavar="max number of features", help="max number of features generated", required=False)
    parser.add_argument("-fw", "--fromweb", type=str, metavar="flag for web", help="flag for web", required=False)
    parser.add_argument("-pm", "--parameter", type=str, metavar="parameters in json", help="json string contains learning alg and parameter selection", required=False)
    parser.add_argument("-pp", "--pca_param", type=str, metavar="pca parameters in json", help="json string contains pca parameter selection", required=False)
    parser.add_argument("-lb", "--lib", type=str, metavar="spark mllib or scikit", help="learning library used", required=False)
    parser.add_argument("-sl", "--showlabelname", type=str, metavar="show label name", help="0: not shown; 1: show label name", required=False)
    parser.add_argument("-dsid", "--ds_id", type=str, metavar="source dataset id", help="source dataset id for training option", required=False)

    parser.add_argument("-ptn", "--pattern_str", type=str, metavar="regular express pattern to extract string"
        , help="regular express pattern to extract string", required=False)
    parser.add_argument("-vb", "--verbose", type=str, metavar="show detailed features", help="show detailed features", required=False)
    parser.add_argument("-ft", "--feat_cnt_threshold", type=str, dest='feat_cnt_threshold', help="feature count to allow prediction"
            , default =config.get('machine_learning', 'feature_count_threshold'))
    
    ###SPARK###
    parser.add_argument('-sp','--sp_master', type=str, dest='sp_master', help='spark.master'
                , default =config.get('spark', 'spark_master'))
    parser.add_argument('-em','--exe_memory', type=str, dest='exe_memory', help='spark.executor.memory'
                , default =config.get('spark', 'spark_executor_memory'))
    parser.add_argument('-cm','--core_max', type=str, dest='core_max', help='spark.cores.max'
                , default =config.get('spark', 'spark_cores_max'))
    
    #### database for output
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
    
    args = parser.parse_args()
    
    if args.name:
        input_gz = args.name
    else:
        input_gz  = '000d9941eaf04efb55e5d0ccff3d90ee.gz'
    if args.out:
        out_dir = args.out
    else:
        out_dir  = '.'
    if args.row_id:
        row_id_str = args.row_id
    else:
        row_id_str  = '553'
    if args.ds_id:
        ds_id = args.ds_id
    else:
        ds_id  = '' 
    if args.cid:
        cid_str = args.cid
    else:
        cid_str  = '01'

    ###################################################
    if args.num:
        num_gram = eval(args.num)
    else:
        num_gram  = eval(config.get("machine_learning","svm_num_gram"))
    if args.para:
        param_in_gram_1 = eval(args.para)
    else:
        param_in_gram_1  = eval(config.get("IN","param_in_gram_1_in"))
    if args.max:
        MAX_FEATURES = eval(args.max)
    else:
        MAX_FEATURES  = eval(config.get("IN","MAX_FEATURES_IN"))
    if args.fromweb:
        fromweb = args.fromweb
    else:
        fromweb  = None        
    if args.parameter:
        j_str = args.parameter
    else:
        j_str='{"c":"1","iterations":"300","regularization":"l2","learning_algorithm":"logistic_regression_with_sgd"}'
    if args.lib: # mllib or scikit
        mode = args.lib
    else:
        mode='scikit'
    if args.showlabelname: # mllib or scikit
        labelnameflag = eval(args.showlabelname)
    else:
        labelnameflag = 0
    if args.verbose: 
        verbose = args.verbose
    else:
        verbose = "1"    
    ######database########################################
    if len(args.username)>0:
        username = args.username
    else:
        username  = None
    if len(args.password)>0:
        password = args.password
    else:
        password  = None     
    
    
    
    t0 = time()
    coef_arr=None
    
    ml_opts = json.loads(j_str)
    
    # ML parameters ===================== input ml_opts ==============
    learning_algorithm=None
    try:
        if ml_opts is None:
            ml_opts = json.loads(j_str)
        learning_algorithm = ml_opts['learning_algorithm']        
    except Exception as e:
        print "WARNING: load learning_algorithm failed.",e
    
    
    # read raw data from .gz file ===================== input .gz ==============
    file_content=None
    try:
        f = gzip.open(input_gz, 'rb')
        file_content = f.read()
        f.close()
    except Exception as e:
        print "ERROR: load data file ["+input_gz+"] failed.",e
        return -5
        
    
    # get data here; assume libsvm format
    label_feature_array=None
    feature_array=None
    label=None
    # optional
    sample_info=None
    if not file_content is None:
        label_features = file_content.strip()
        if not label_features is None:
            label_feature_array = label_features.split(' ')
            label = label_feature_array[0]
            # check if 1st item is integer
            int_1st="y"
            try:
                int(label_feature_array[0])
            except ValueError:
                int_1st="n"
            # check if 2nd item is integer
            int_2nd="y"
            try:
                int(label_feature_array[0])
            except ValueError: 
                int_2nd="n"
                
            if int_1st=="n" and int_2nd=="y":
                feature_array = label_feature_array[2:len(label_feature_array)]
            elif int_1st=="y" and int_2nd=="n":
                feature_array = label_feature_array[1:len(label_feature_array)]
                
            #feature_array = label_feature_array[1:len(label_feature_array)]
            # if sample_info exists, 2nd item will be digit and 3rd item won't be digit 
            if not label.isdigit() and label_feature_array[1].isdigit() and not label_feature_array[2].isdigit():
                sample_info=label
                label=label_feature_array[1]
                feature_array=label_feature_array[2:len(label_feature_array)] 
        else:
            print "ERROR: data format error!"
    else:
        print "ERROR: no data found!"
    #print label  ### -1 means no label
    #print feature_array
    curr_dic = {}
    #print "feature_array=",feature_array
    for features in feature_array:
        if len(features)>0:
            key, value = features.split(':')
            curr_dic[key] = float(value)
        else:
            print "WARNING: data format error!"
            
    print "INFO: curr_dic len=",len(curr_dic)

    if curr_dic and verbose=="1":

        #print "INFO: *** Feature list: ===================================="
        # clean up feature file
        out_file=os.path.join(out_dir,cid_str+"_feature_list.json")
        print "INFO: feature file=",out_file
        if os.path.exists(out_file):
            try:
                os.remove(out_file)
            except OSError, e:
                print ("ERROR: %s - %s." % (e.strerror, out_file))
        out_f=open(out_file, 'a')   
        
        # get coef_arr ==================================
        if coef_arr is None and not learning_algorithm in ('kmeans'):
            key = "coef_arr"  #{"123":"openFile"}
            jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
            jstr_proj='{"value":1}'
            # each model has its own coef_arr
                
            doc=query_mongo.find_one(args.ip_address, args.port, args.db_name, args.tb_name, username, password, jstr_filter, jstr_proj)
            coef_arr = doc['value']
        
        fout_arr=[]
        len_coef=len(coef_arr)
        for k,v in curr_dic.items():
            feat_out={}
            feat_out["ngram"]=""
            if int(k) < len_coef:
                feat_out["fid"]=k
                feat_out["coef"]=coef_arr[int(k)-1]
                feat_out["desc"]=""
            else:
                feat_out["fid"]="None"
                feat_out["coef"]=0
                feat_out["desc"]=str(k)
                
            fout_arr.append(feat_out)

        if len(fout_arr) > 0:
            out_f.write(json.dumps(fout_arr))               
        out_f.close()

            
    if mode == "scikit": #"SKlean":
        # get the ML model
        model_file  = out_dir + '/' + row_id_str + '_model/' + row_id_str + '.pkl'

        # load clf from model file
        sk_model = joblib.load(model_file)
        clf_coef_arr=None
        intercept_arr=None
        
        print "INFO: model type=",type(sk_model)," clf=",sk_model
        print "INFO: clf=",sk_model
        try:
            if type(sk_model) in ( classes.SVC , classes.NuSVC) :# svm didn't have coef_
                col_num=sk_model.support_vectors_.shape[1]
            elif learning_algorithm in ('kmeans') :
                print "INFO: Kmeans cluster_centers_ =", sk_model.cluster_centers_ 
                # to convert sample to np array
                col_num=dic_len
            else: #linear only
                col_num = len(sk_model.coef_[0])
                clf_coef_arr=sk_model.coef_[0]
                intercept_arr=sk_model.intercept_
            # coef_ is only available when using a linear kernel
        except Exception as e:
            print "WARNING: Can't get sk_model.coef_[0]. e=",e,", get total features from meta-data"
            col_num = dic_len #how to get feature number for sparse array? 
        print "INFO: total feature # in sklearn model=", col_num

        
        # generate the sparse matrix from new_curr_dic
        #sparse_test = matrix_gen_from_dic(curr_dic, col_num)
        sparse_test = ml_util.generate_matrix_from_dic(curr_dic, col_num)

        labels_pred = sk_model.predict(sparse_test)
        
        sing_label_pred = labels_pred[0]
    
         # calculate hypothesis value ================
        if not clf_coef_arr is None and not intercept_arr is None:
            clf_classname=sk_model.__class__.__name__
            predict_val=ml_util.calculate_hypothesis(curr_dic, col_num, clf_coef_arr, intercept_arr[0], clf_classname)
            print "INFO: intercept_arr[0]=",intercept_arr[0], ", len=", len(intercept_arr)
            #??print "--threshold=",sk_model.threshold
            print "INFO: clf_coef_arr size=",clf_coef_arr.size
            print "INFO: clf_classname=",clf_classname,", h(wx)=",predict_val
   
    else: # spark mllib
        ####pyspark#####: to allow python to run this code. pyspark is only available by spark-submit 
        from pyspark import SparkContext
        from pyspark.sql import SQLContext
        from pyspark.mllib.util import MLUtils
        from pyspark.mllib.regression import LabeledPoint
        from pyspark.mllib.classification import SVMWithSGD, SVMModel
        from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionWithSGD, LogisticRegressionModel
        from pyspark.mllib.linalg import SparseVector
        from pyspark.mllib.evaluation import BinaryClassificationMetrics
        from pyspark.mllib.tree import DecisionTree
        from pyspark.mllib.clustering import KMeans, KMeansModel, GaussianMixture, GaussianMixtureModel
        from pyspark.mllib.linalg import Vectors 
        
        SparkContext.setSystemProperty('spark.rdd.compress', config.get('spark', 'spark_rdd_compress'))
        SparkContext.setSystemProperty('spark.driver.maxResultSize', config.get('spark', 'spark_driver_maxResultSize'))
        SparkContext.setSystemProperty('spark.executor.memory', args.exe_memory)
        SparkContext.setSystemProperty('spark.cores.max', args.core_max)

        sc = SparkContext(args.sp_master, 'single_predict:'+str(args.row_id))
        flag_model = ml_opts['learning_algorithm']        
        save_dir = config.get('app', 'HADOOP_MASTER')+config.get('app', 'HDFS_MODEL_DIR')+'/'+row_id_str

        if flag_model == "linear_svm_with_sgd":
            mllib_model = SVMModel.load(sc, save_dir)
            col_num = len(mllib_model.weights)
        elif flag_model == "logistic_regression_with_lbfgs" or flag_model == "logistic_regression_with_sgd":
            mllib_model = LogisticRegressionModel.load(sc, save_dir)
            col_num = mllib_model.numFeatures # len(mllib_model.weights) return 3x value
        elif flag_model == "kmeans":
            mllib_model = KMeansModel.load(sc, save_dir)
            col_num =len(mllib_model.clusterCenters[0])
        else:
            print "ERROR: Training model selection error: no valid ML model selected!"
            return
        # get the model dimension
        #col_num = len(mllib_model.weights)
        print "INFO: total feature # in mllib model=",col_num

        # calculate hypothesis value ================
        model_weight=None
        if learning_algorithm not in ("kmeans") :
            model_weight=mllib_model.weights
            intercept=mllib_model.intercept 

        coef_arr=None
        predict_val=None    
        if not model_weight is None :
            coef_arr=np.asarray(model_weight.toArray().tolist())

        if not coef_arr is None and not intercept is None:
            classname=mllib_model.__class__.__name__
            print "coef_arr=",coef_arr
            try:
                predict_val=ml_util.calculate_hypothesis(curr_dic, col_num, coef_arr, intercept, classname)
            except Exception as e:
                print "WARNING: calculate_hypothesis failed. e=",e
                
            print "INFO: mllib_model.intercept=",intercept
            print "INFO: mllib_model.threshold=",mllib_model.threshold
            print "INFO: coef_arr size=",coef_arr.size
            print "INFO: classname=",classname,", h(wx)=",predict_val

            
        # generate the sparse matrix from new_curr_dic
        #vector_test = vector_gen_from_dic(curr_dic, col_num)
        vector_test = ml_util.generate_vector_from_dic(curr_dic, col_num)
        # use API to generate vector
        #vector_test=SparseVector(col_num,curr_dic)
        #print "INFO: vector_test size=",len(vector_test)
        
        sing_label_pred = mllib_model.predict(vector_test)         
        

    print "RESULT: predict output=", sing_label_pred      

    ### generate label names (family names) #####
    ### connect to database to get the column list which contains all column number of the corresponding feature####
    pred_label=None
    if labelnameflag == 1:
        key = "dic_name_label"
        jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
        jstr_proj='{"value":1}'

        # get parent dataset's data
        if ds_id != row_id_str:
            jstr_filter='{"rid":'+ds_id+',"key":"'+key+'"}'
        
        doc=query_mongo.find_one(args.ip_address, args.port, args.db_name, args.tb_name, username, password, jstr_filter, jstr_proj)
        dic_list = doc['value']
        
        label_dic = {}
        for i in range(0, len(dic_list)):
            for key in dic_list[i]:
                label_dic[dic_list[i][key]] = key.encode('UTF8')
        print "INFO: label_dic:", label_dic
        try:
            pred_label = label_dic[int(sing_label_pred)]
        except Exception as e:
            print "WARNING: Can't get label",e
            pred_label=str(sing_label_pred)
    else:
        pred_label = str(sing_label_pred)
        print "RESULT: prediction=", pred_label
    
    ###################################
    ############update DB##############
    ###################################
    
    # only update db for web request
    if fromweb=="1": 
        #print "database update"
        str_sql="UPDATE atdml_document set status = 'predicted', processed_date ='" \
            +str(datetime.datetime.now())+"', prediction = '"+ pred_label  \
            +"', predict_val = '"+str(predict_val) \
            +"' where id="+cid_str
        ret=exec_sqlite.exec_sql(str_sql)
        #print "Data update done! ret=", str(ret)
    else:
        print "RESULT: prediction="+ pred_label+""
    
    
    t1 = time()
    print 'INFO: total running time: %f' %(t1-t0)
    return 0


if __name__ == '__main__':
    __description__ = "single file prediction for generic"
    main()
