#! /usr/bin/python
'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''
#
# python libraries
import os
import os.path, glob
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
from collections import Counter

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
from sklearn.svm import NuSVC, classes, LinearSVC
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
sys.path.append('./db')
import query_mongo
import ml_util
from ml_util import *

# for generating feature list
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
    parser.add_argument("-mf", "--modelfolder", type=str, metavar="model folder", help="model for prediction", required=False)
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
    #if args.name:
    #    file_name_given = args.name
    #else:
    #    file_name_given  = 'aaaa'
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
    if args.modelfolder:
        model_data_folder = args.modelfolder
    else:
        model_data_folder  = local_out_dir + '/' + row_id_str + '_model/'
    if args.fromweb:
        fromweb = args.fromweb
    else:
        fromweb  = None
    if args.parameter:
        ml_opts_jstr = args.parameter
    else:
        ml_opts_jstr  = '{"learning_algorithm":"logistic_regression", "cv":"3", "mode":"cheap", "api":"centralized"}'
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
        zip_file_name  = 'train_sklearn_cv.zip'

    # mongo info for connection
    mongo_tuples=(args.ip_address, args.port, args.db_name, args.tb_name, username, password)
    
    return train(row_id_str, ds_id, hdfs_feat_dir, local_out_dir, ml_opts_jstr, excluded_feat_cslist
    , args.sp_master,config.get('spark', 'spark_rdd_compress'),config.get('spark', 'spark_driver_maxResultSize'), args.exe_memory, args.core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples, labelnameflag, fromweb
    , training_fraction, 'train_grid_search_scikit:'+row_id_str, model_data_folder )


# ================================================================================== train () ============ 
def train(row_id_str, ds_id, hdfs_feat_dir, local_out_dir, ml_opts_jstr, excluded_feat_cslist
    , sp_master, spark_rdd_compress, spark_driver_maxResultSize, sp_exe_memory, sp_core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples, labelnameflag, fromweb
    , training_fraction, jobname, model_data_folder ): 
    

    # zip func in other files for Spark workers ================= ================
    zip_file_path = ml_build_zip_file(zipout_dir, zipcode_dir, zip_file_name, prefix='zip_feature_util')
    print "INFO: zip_file_path=",zip_file_path
    

    # ML model filename ====
    model_fname=os.path.join(model_data_folder, row_id_str+'.pkl')
    print "INFO: model_data_folder=",model_data_folder    
    # create out folders and clean up old model files ====
    ml_util.ml_prepare_output_dirs(row_id_str,local_out_dir,model_data_folder,model_fname)   

    # init Spark context ====
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
    if not ml_opts_jstr is None:
        ml_opts=json.loads(ml_opts_jstr)
        if "has_excluded_feat" in ml_opts:
            has_excluded_feat=ml_opts["has_excluded_feat"]

    # get excluded feature list from mongo ========== ===
    if str(has_excluded_feat) == "1" and excluded_feat_cslist is None:
        excluded_feat_cslist=ml_util.ml_get_excluded_feat(row_id_str, mongo_tuples)
    print "INFO: excluded_feat_cslist=",excluded_feat_cslist
            
    # source libsvm filename  
    libsvm_data_file = os.path.join(hdfs_feat_dir , "libsvm_data")
    print "INFO: libsvm_data_file=", libsvm_data_file

    # load feature count file
    feat_count_file=libsvm_data_file+"_feat_count"
    feature_count=zip_feature_util.get_feature_count(sc,feat_count_file)
    print "INFO: feature_count=",feature_count

    
    # load sample RDD from text file   
    #   also exclude selected features in sample ================ =====
    # format (LabeledPoint,hash) from str2LabeledPoint_hash() 
    #samples_rdd = MLUtils.loadLibSVMFile(sc, libsvm_data_file)
    samples_rdd,feature_count = zip_feature_util.get_sample_rdd(sc, libsvm_data_file, feature_count, excluded_feat_cslist)

    all_data = samples_rdd.collect()
    sample_count=len(all_data)
    # 2-D array
    features_list = [x.features.toArray() for x,_ in all_data]
    # label array
    labels_list_all = [x.label for x,_ in all_data]
    # hash array
    hash_list_all = [x for _,x in all_data]

    # convert to np array
    labels_list_all = array(labels_list_all)
    features_array = np.array(features_list)
    hash_list_all=np.array(hash_list_all)
    
    # generate sparse matrix (csr) for all samples
    features_sparse_mtx = csr_matrix(features_array)

    ### randomly split the samples into training and testing data ===============
    X_train_sparse, X_test_sparse, labels_train, labels_test, train_hash_list, test_hash_list = \
            cross_validation.train_test_split(features_sparse_mtx, labels_list_all, hash_list_all, test_size=(1-training_fraction) )
    # X_test_sparse is scipy.sparse.csr.csr_matrix
    testing_sample_count = len(labels_test)
    training_sample_count=len(labels_train)
    training_lbl_cnt_list=Counter(labels_train)
    testing_lbl_cnt_list=Counter(labels_test)
    
    print "INFO: training sample count=",training_sample_count,", testing sample count=",testing_sample_count,",sample_count=",sample_count
    print "INFO: training label list=",training_lbl_cnt_list,", testing label list=",testing_lbl_cnt_list
    print "INFO: train_hash_list count=",len(train_hash_list),", test_hash_list count=",len(test_hash_list)
    t1 = time()
    print 'INFO: running time: %f' %(t1-t0)
    
    ###############################################
    ###########build learning model################
    ###############################################
    
    ### parse parameters and generate the model ###
    (clf, model_name, api, cv, param_dic) = parse_param_and_get_model(ml_opts)
    if model_name == "none":
        print "ERROR: model name not found!"
        return -1

    #param_jobj=json.loads(ml_opts_jstr);
    #print "param_jobj=",param_jobj
        
    ########################################################
    ##########Grid Search with cross validation#############
    ########################################################    
    json2save={}
    json2save["rid"]=int(row_id_str)
    json2save["key"]="cv_result"
    #json2save["param_str"]=ml_opts_jstr
    json2save["param_dic"]=param_dic
    cv_grid=[]
    if api == "centralized":
        #########run with Scikit-learn API (for comparison)######
        print "INFO: ******************Grid Search with Scikit-learn API************"

        t0 = time()
        
        # Set the parameters by cross-validation
        #tuned_parameters = [{'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}]
        #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], \
        #                 'C': [1, 10, 100, 1000]}, \
        #                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

        scores = ['accuracy']
        json2save["scores"]=scores
        #print json2save
        
        for score in scores: # for one item only? score=accuracy
            print("INFO: # Tuning hyper-parameters for %s" % score)
            #print()

            grid = grid_search.GridSearchCV(estimator = clf, param_grid = param_dic, cv=cv, scoring= score)
            grid.fit(X_train_sparse, labels_train)
            
            print "INFO: Best parameters set found on development set:"
            print "INFO: grid.best_params_=",grid.best_params_
            print "INFO: Grid scores on development set:" 
            for key in grid.best_params_:
                print "INFO: best_params["+key+"]=", grid.best_params_[key]
                if key.lower()=="regtype":
                    ml_opts['regularization']=str(grid.best_params_[key]) # add best param to 
                else:
                    ml_opts[key.lower()]=str(grid.best_params_[key]) # add best param to 
            # save best param to db as json string
            j_str=json.dumps(ml_opts);
            json2save["param_str"]=j_str;
            print "INFO: grid_scores_ with params:"
            for params, mean_score, scores in grid.grid_scores_:
                print "INFO: %0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params)
                #outstr='%s,%0.3f,%0.03f,%s' % (params,mean_score, scores.std() * 2,"Selected" if params==grid.best_params_ else "")
                outj={}
                outj["param"]=params
                outj["average_accuracy"]="%0.3f" % (mean_score)
                outj["std_deviation"]="%0.3f" % (scores.std() * 2)
                outj["selected"]="%s" % ("Selected" if params==grid.best_params_ else "")
                
                cv_grid.append(outj)
        
        clf_best = grid.best_estimator_
        t1 = time()
        ############# END run with SKlearn ######
        print 'INFO: Grid Search with SKlearn running time: %f' %(t1-t0)
        t0 = time()
    else:
    
        #############run with SPARK######
        
        print "INFO: ******************Grid Search with SPARK************"
            
        all_comb_list_of_dic = get_all_combination_list_of_dic(param_dic) 
        print "INFO: Total number of searching combinations=", len(all_comb_list_of_dic) 
        #print "all_comb_list_of_dic: ", all_comb_list_of_dic
        params_rdd = sc.parallelize(all_comb_list_of_dic)
        
        ###broad cast clf, traning data, testing data to all workers###
        X_broadcast = sc.broadcast(X_train_sparse)
        y_broadcast = sc.broadcast(labels_train)
        clf_broadcast = sc.broadcast(clf)
        
        ### Grid Search with CV in multiple workers ###
        models = params_rdd.map(lambda x: learn_with_params(clf_broadcast.value, X_broadcast.value, y_broadcast.value, cv, x)).sortByKey(ascending = False).cache()
        
        (ave_accuracy, (clf_best, p_dic_best, std2))  = models.first()
        # output results #

        print "INFO: Best parameters set found for ", model_name, " is: "
        print "INFO: ",
        for key in p_dic_best:
            print key, " = ", p_dic_best[key],
            if key.lower()=="regtype":
                ml_opts['regularization']=str(p_dic_best[key]) 
            else:
                ml_opts[key.lower()]=str(p_dic_best[key]) # add best param to 
            # save best param to db as json string
        print ""
        j_str=json.dumps(ml_opts);
        json2save["param_str"]=j_str;

        print "INFO: Average accuracy with CV = ", cv, ": ", ave_accuracy
        
        ######## print complete report #######
        print "INFO: Grid scores on development set:"
        all_results = models.collect()
        for i in range(0, len(all_results)):
            (ave_accu_i, (clf_i, p_dic_i, std2_i)) = all_results[i]
            print "INFO: ",ave_accu_i, " for ", p_dic_i
            print "INFO: %0.3f (+/-%0.03f) for " % (ave_accu_i, std2_i), p_dic_i
            #outstr='%s,%0.3f,%0.03f,%s' % ( p_dic_i, ave_accu_i, std2_i, "Selected" if p_dic_i==p_dic_best else "")
            outj={}
            outj["param"]=p_dic_i
            outj["average_accuracy"]="%0.3f" % (ave_accu_i)
            outj["std_deviation"]="%0.3f" % (std2_i)
            outj["selected"]="%s" % ("Selected" if p_dic_i==p_dic_best else "")
            
            cv_grid.append(outj)
        print " "
        
        t1 = time()
        
        ############# END run with SPARK######
        print 'INFO: Grid search with SPARK running time: %f' %(t1-t0)
    
    ##################################################################################
    #print "cv_grid=",cv_grid
    #json2save["cv_grid_title"]='param,average_accuracy,std_deviation,selected' 
    json2save["cv_grid_data"]=cv_grid
    json2save['clf_best']=str(clf_best).replace("\n","").replace("    ","")
    cv_result=json.dumps(json2save)
    #print "INFO: cv_result=",cv_result
    filter='{"rid":'+row_id_str+',"key":"cv_result"}'
    upsert_flag=True
    ## write to mongoDB.myml.dataset_info, ignore doc with duplicated key
    # db.dataset_info.createIndex({"rid":1,"key":1},{unique:true})
    ret=query_mongo.upsert_doc_t(mongo_tuples,filter,cv_result,upsert_flag)
    print "INFO: Upsert count for cv_result: ret=",ret
 
    ##################################################################################
    ##########Retrain with best model for training set and output results#############
    ##################################################################################
    print "INFO: **********Retrain with best model for training set and output results************"
    
    clf_best.fit(X_train_sparse, labels_train)
    #### save clf_best for future use ####
    #joblib.dump(clf_best, model_data_folder + row_id_str+'.pkl')
    joblib.dump(clf_best, model_fname) 
    
    ### Evaluating the model on testing data
    labels_pred = clf_best.predict(X_test_sparse)
    accuracy = clf_best.score(X_test_sparse, labels_test)
    print "INFO: Accuracy = ", accuracy
    
    
    ######################################the rest of the code is the same as train_sklean.py (replace clf with clf_best)#####################################################################
    clf=clf_best
    print "INFO: model type=",type(clf)," clf=",clf

    # get data from model ================================
    coef=None
    intercept=None
    try:
        if type(clf) in ( classes.SVC , classes.NuSVC) :# svm didn't have coef_
            col_num=clf.support_vectors_.shape[1]
        else: #linear only
            # coef_ is only available when using a linear kernel
            col_num = len(clf.coef_[0])
            coef=clf.coef_[0]
            intercept=clf.intercept_[0] # only get 1st item?
            #print "**model:clf.coef_[0] =",clf.coef_[0]
    except Exception as e:
        print "WARNING: Can't get clf.coef_[0]. e=",e,", get total features from meta-data"
        col_num = 0 #how to get feature number for sparse array? 
    print "INFO: total feature # in the model: ", col_num

    jfeat_coef_dict={}
    # create feature coefficient file ================================
    if coef is None:
        print "WARNING: model weights not found!"    
    else:
        feat_filename=os.path.join(local_out_dir,row_id_str+"_feat_coef.json")
        print "INFO: feat_filename=",feat_filename
        # save coef_arr to mongo & create jfeat_coef_dict===
        jfeat_coef_dict=ml_util.ml_save_coef_build_feat_coef(row_id_str, mongo_tuples, coef, intercept, feat_filename, ds_id)
    #print "INFO: jfeat_coef_dict=", jfeat_coef_dict
    print "INFO: jfeat_coef_dict len=", len(jfeat_coef_dict )


    # filename for false pred 
    false_pred_fname=os.path.join(local_out_dir,row_id_str+"_false_pred.json")
    print "INFO: false_pred_fname=", false_pred_fname

    # build files for false pred & score graph
    (score_arr_0, score_arr_1, max_score,min_score)=ml_build_false_pred(X_test_sparse,coef,intercept
        , labels_test, labels_pred, test_hash_list, model_name, jfeat_coef_dict, false_pred_fname) 

    # save pred output
    pred_out_arr=[]
    for i in range(0,len(labels_test)):
        pred_out_arr.append((labels_test[i], labels_pred[i], test_hash_list[i]))
    pred_ofname=os.path.join(local_out_dir,row_id_str+"_pred_output.pkl")
    print "INFO: pred_ofname=", pred_ofname
    ml_util.ml_pickle_save(pred_out_arr,pred_ofname)
    
    ###################################################
    ### generate label names (family names) ###########
    ### connect to database to get the column list which contains all column number of the corresponding feature####
    ###################################################
    
    if labelnameflag == 1:
        key = "dic_name_label"
        jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
        jstr_proj='{"value":1}'

        # get parent dataset's data
        if ds_id != row_id_str:
            jstr_filter='{"rid":'+ds_id+',"key":"'+key+'"}'
        
        doc=query_mongo.find_one_t(mongo_tuples, jstr_filter, jstr_proj)
        dic_list = doc['value']
        
        label_dic = {}
        for i in range(0, len(dic_list)):
            for key in dic_list[i]:
                label_dic[dic_list[i][key]] = key.encode('UTF8')
        print "INFO: label_dic:", label_dic
    else:
        label_dic = {}
        label_set = set(labels_list_all)
        for label_value in label_set:
            label_dic[int(label_value)] = str(int(label_value))
        print "INFO: ******generated label_dic:", label_dic 
    
    labels_list = []
    for key in sorted(label_dic):
        labels_list.append(label_dic[key])
    
    ### generate sample numbers of each family in testing data###
    testing_sample_number = len(labels_test)
    print "INFO: testing_sample_number=", testing_sample_number
    test_cnt_dic = {}
    for key in label_dic:
        test_cnt_dic[key] = 0
    for i in range (0, testing_sample_number):
        for key in label_dic:
            if labels_test[i] == key:
                test_cnt_dic[key] = test_cnt_dic[key] + 1
    print "INFO: Number of samples in each label is=", test_cnt_dic
    
    ###############################################
    ###########plot prediction result figure#######
    ###############################################
    pred_fname=os.path.join(local_out_dir,row_id_str+"_1"+".png")
    true_fname=os.path.join(local_out_dir,row_id_str+"_2"+".png")
    pred_xlabel='Prediction (Single Run)'
    true_xlabel='True Labels (Single Run)'
    test_cnt_dic=ml_util.ml_plot_predict_figures(labels_pred.tolist(), labels_test.tolist(), labels_list, label_dic, testing_sample_count 
        , pred_xlabel, pred_fname, true_xlabel, true_fname)
    print "INFO: figure files: ", pred_fname, true_fname
    print "INFO: Number of samples in each label is=", test_cnt_dic

    roc_auc=None
    #fscore=None 
    perf_measures=None
    class_count=len(labels_list)
    dataset_info={"training_fraction":training_fraction, "class_count":class_count,"dataset_count":sample_count}
    #############################################################
    ###################for 2 class only (plot ROC curve)#########
    #############################################################
    if len(labels_list) == 2:

        # build data file for score graph
        score_graph_fname=os.path.join(local_out_dir,row_id_str+"_score_graph.json")
        print "INFO: score_graph_fname=", score_graph_fname
        ml_build_pred_score_graph(score_arr_0,score_arr_1,model_name, score_graph_fname,max_score,min_score)

            
        do_ROC=True
        reverse_label_dic = dict((v,k) for k, v in label_dic.items())
        if 'clean' in reverse_label_dic:
            flag_clean = reverse_label_dic['clean']
        elif 'benign' in reverse_label_dic:
            flag_clean = reverse_label_dic['benign']
        elif '0' in reverse_label_dic:
            flag_clean = 0
        else:
            print "No ROC curve generated: 'clean' or '0' must be a label for indicating negative class!"
            do_ROC=False
            
        if do_ROC:
            # calculate fscore  ==========
            perf_measures=ml_util.calculate_fscore(labels_test, labels_pred)
            print "INFO: perf_measures=",perf_measures
            
            confidence_score = clf_best.decision_function(X_test_sparse)
                    
            if flag_clean == 0:
                scores = [x for x in confidence_score]
                s_labels = [x for x in labels_test]
                testing_N = test_cnt_dic[0]
                testing_P = test_cnt_dic[1]
            else:
                scores = [-x for x in confidence_score]
                s_labels = [1-x for x in labels_test]
                testing_N = test_cnt_dic[1]
                testing_P = test_cnt_dic[0]
                
            # create ROC data file ======== ==== 
            roc_auc=ml_create_roc_files(row_id_str, scores, s_labels, testing_N, testing_P
                , local_out_dir, row_id_str)
                
            perf_measures["roc_auc"]=roc_auc
            
                
    # only update db for web request
    if fromweb=="1": 
        #print "database update"
        str_sql="UPDATE atdml_document set "+"accuracy = '"+str(accuracy*100)+"%" \
            +"', status = 'learned', processed_date ='"+str(datetime.datetime.now()) \
            +"',ml_opts='"+j_str \
            +"', perf_measures='"+json.dumps(perf_measures) \
            +"', dataset_info='"+json.dumps(dataset_info) \
            +"' where id="+row_id_str
        ret=exec_sqlite.exec_sql(str_sql)
        print "INFO: Data update done! ret=", str(ret)
    else:
        print "INFO: accuracy = '"+str(accuracy*100)+"%"
    
    print 'INFO: total running time: %f' %(t1-t00)
    
    print 'INFO: Finished!'
    return 0

def learn_with_params(clf_in, sparse_X, labels_y, cv, x):
    
    clf = clf_in
    p_dic = x
    clf.set_params(**p_dic)
    scores = cross_validation.cross_val_score(clf, sparse_X, labels_y, cv=cv)
    ave_accuracy = scores.mean()
    std2 = scores.std() * 2
    return (ave_accuracy, (clf, p_dic, std2))

def get_all_combination_list_of_dic(param_dic):
    all_comb_list_of_dic = []
    for p in range (0, len(param_dic)):
        dic = param_dic[p]
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

def parse_param_and_get_model(param_dict):
    
    #param_dict = json.loads(j_str)
    model_name = param_dict['learning_algorithm']     # 1: linear_svm; 2: ; 3: 
    cv = eval(param_dict['cv'])
    mode = param_dict['mode']
    api = param_dict['api']
    
    print "INFO: Learning Algorithm: ", model_name
    print "INFO: CV = ", cv
    print "INFO: mode = ", mode
    print "INFO: API use: ", api
    ###parse and print print parameters###
    print "INFO: ============ Learning Algorithm and Grid Search Parameters ============="    
    
    if model_name == "linear_svm":
        ### 1: linearSVM
        if mode == "cheap":
            param_dic = [{'C': [0.0001, 0.01, 1, 100, 10000]}]
        else:
            param_dic = [{'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}]
        print "INFO: Grid Search Parameters:"
        print "INFO: C = ", param_dic[0]['C']                       
        print "INFO: ====================1: Linear SVM============="
        clf = svm.LinearSVC()
    elif model_name == "svm":
        ### 2: SVM with kernel
        if mode == "cheap":
            param_dic = [{'C': [0.01, 1, 100], 'kernel':['rbf','sigmoid'], 'gamma':[0.0, 0.5]}, {'C': [0.01, 1, 100], 'kernel':['linear']}, {'C': [0.01, 1, 100], 'kernel':['poly'], 'gamma':[0.0, 0.5], 'degree':[3]}]
        else:
            param_dic = [{'C': [0.0001, 0.01, 1, 100, 10000], 'kernel':['rbf','sigmoid'], 'gamma':[0.0, 0.5, 1]}, {'C': [0.0001, 0.01, 1, 100, 10000], 'kernel':['linear']}, {'C': [0.0001, 0.01, 1, 100, 10000], 'kernel':['poly'], 'gamma':[0.0, 0.5], 'degree':[2,3]}]
            #param_dic = [{'C': [0.0001, 0.01, 1, 100, 10000], 'kernel':['rbf','sigmoid'], 'gamma':[0.0, 0.5, 1]}, {'C': [0.0001, 0.01, 1, 100, 10000], 'kernel':['linear']}, {'C': [0.0001, 0.01, 1, 100, 10000], 'kernel':['poly'], 'gamma':[0.0, 0.5, 1], 'degree':[2,3]}]
            #param_dic = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel':['rbf','sigmoid'], 'gamma':[0.0, 0.01, 0.1, 1, 10, 100]}, {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel':['linear']}, {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel':['poly'], 'gamma':[0.0, 0.01, 0.1, 1, 10, 100], 'degree':[2,3,4]}]            
        print "INFO: Grid Search Parameters:"
        for p in range (0, len(param_dic)):
            print "INFO: ",
            for key in param_dic[p]:
                print key, ' = ', param_dic[p][key],
            print ""
        print "INFO: ====================2: SVM with kernel============="
        clf = svm.SVC()
    elif model_name == "nu_svm":
        ### 3: NuSVC
        if mode == "cheap":
            param_dic = [{'nu': [0.1, 0.3], 'kernel':['rbf','sigmoid'], 'gamma':[0.0, 0.1]}, {'nu': [0.1, 0.3], 'kernel':['linear']}, {'nu': [0.1, 0.3], 'kernel':['poly'], 'gamma':[0.0, 0.1], 'degree':[3]}]
        else:
            param_dic = [{'nu': [0.1, 0.2, 0.3], 'kernel':['rbf','sigmoid'], 'gamma':[0.0, 0.1, 1, 10]}, {'nu': [0.1, 0.2, 0.3], 'kernel':['linear']}, {'nu': [0.1, 0.2, 0.3], 'kernel':['poly'], 'gamma':[0.0, 0.1, 1, 10], 'degree':[2,3]}]
            #param_dic = [{'nu': [0.1, 0.2, 0.3, 0.4], 'kernel':['rbf','sigmoid'], 'gamma':[0.0, 0.1, 1, 10]}, {'nu': [0.1, 0.2, 0.3, 0.4], 'kernel':['linear']}, {'nu': [0.1, 0.2, 0.3, 0.4], 'kernel':['poly'], 'gamma':[0.0, 0.1, 1, 10], 'degree':[2,3]}]            
        print "INFO: Grid Search Parameters:"
        for p in range (0, len(param_dic)):
            print "INFO: ",
            for key in param_dic[p]:
                print key, ' = ', param_dic[p][key],
            print ""
        print "INFO: ====================3: NuSVC============="
        clf = svm.NuSVC()
    elif model_name == "logistic_regression":
        ### 4: Logistic Regression
        if mode == "cheap":
            param_dic = [{'C': [0.0001, 0.01, 1, 100, 10000], 'penalty':['l2']}]
        else:
            param_dic = [{'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'penalty':['l2', 'l1']}]
        print "INFO: Grid Search Parameters:"
        print "INFO: C= ", param_dic[0]['C']
        print "INFO: penalty= ", param_dic[0]['penalty']                
        print "INFO: ====================4: Logistic Regression============="
        clf = linear_model.LogisticRegression()
    elif model_name == "passive_aggressive_classifier":
        ### 6: Passive Aggressive Classifier
        if mode == "cheap":
            param_dic = [{'C': [0.0001, 0.01, 1, 100, 10000]}]
        else:
            param_dic = [{'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}]
        print "INFO: Grid Search Parameters:"
        print "INFO: C= ", param_dic[0]['C']       
        print "INFO: ====================6: Passive Aggressive Classifier============="
        clf = linear_model.PassiveAggressiveClassifier()
    else:
        print "INFO: Training model selection error: no valid ML model selected!"
        return (0, "none", 0, 0, 0)
    return (clf, model_name, api, cv, param_dic)
''' move to zip_feature_util
# get input data RDD in LabeledPoint format; may exclude some features     
def get_sample_data_from_hdfs(row_id_str,sc,libsvm_data_file, has_excluded_feat=0,excluded_feat_cslist=None
        , mongo_tuples=None):    
            
    # get excluded feature list from mongo ========== ===
    if str(has_excluded_feat) == "1" and excluded_feat_cslist is None and not mongo_tuples is None:
        key = "feature_excluded"
        jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
        jstr_proj='{"value":1}'
        
        # not get parent dataset's data
        doc=query_mongo.find_one_t(mongo_tuples, jstr_filter, jstr_proj)
        if not doc is None and 'value' in doc:
            excluded_feat_cslist = ','.join(str(i) for i in doc['value'])
    print "INFO: excluded_feat_cslist=",excluded_feat_cslist

    
    samples_rdd = MLUtils.loadLibSVMFile(sc, libsvm_data_file)
    #print samples_rdd.count() 
    
    # Exclude selected features in sample ================ =====
    if excluded_feat_cslist is None:
        labels_and_features_rdd = samples_rdd.map(lambda p: (p.label, p.features))
    else:
        labels_and_features_rdd = samples_rdd \
            .map(lambda p: (p.label, p.features)) \
            .map(lambda p: exclude_feature(p[0], p[1].size, p[1].indices, p[1].values, excluded_feat_cslist)) \
            .map(lambda p: LabeledPoint(p[0], SparseVector(p[1], p[2], p[3])) ) \
            .map(lambda p: (p.label, p.features)) \
    
    # return RDD of LabeledPoint
    return labels_and_features_rdd
'''    
if __name__ == '__main__':
    __description__ = "Grid Search for ML single run with CV"
    main()
