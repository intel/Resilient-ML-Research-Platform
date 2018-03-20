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
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import load_svmlight_file
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

#####import for django database####
sys.path.append('./db')
import exec_sqlite

####import our own library####
sys.path.append('./db')
import query_mongo

CONF_FILE='../../app.config' # at the base dir of the web
config=ConfigParser.ConfigParser()
config.read(CONF_FILE)
portion = eval(config.get("machine_learning","portion"))
mtx_name_list = config.get("machine_learning","mtx_name_list")
mtx_libsvm = config.get("machine_learning","mtx_libsvm")
mtx_stat = config.get("machine_learning","mtx_stat")

flag_model = "linear_svm_with_sgd"

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
    parser.add_argument("-mf", "--modelfolder", type=str, metavar="model folder", help="model for prediction", required=False)
    parser.add_argument("-l", "--listfile", type=str, metavar="list file", help="list of testing data hashes for single file prediction", required=False)
    parser.add_argument("-u", "--uploadtype", type=str, metavar="upload type", help="data type", required=False)
    parser.add_argument("-w", "--fromweb", type=str, metavar="flag for web", help="flag for web", required=False)
    parser.add_argument("-pm", "--parameter", type=str, metavar="parameters in json", help="json string contains learning alg and parameter selection", required=False)

    parser.add_argument('-sp','--sp_master', type=str, dest='sp_master', help='spark.master'
                , default =config.get('spark', 'spark_master'))
    parser.add_argument('-em','--exe_memory', type=str, dest='exe_memory', help='spark.executor.memory'
                , default =config.get('spark', 'spark_executor_memory'))
    parser.add_argument('-cm','--core_max', type=str, dest='core_max', help='spark.cores.max'
                , default =config.get('spark', 'spark_cores_max'))

    #### database
    parser.add_argument('-ip','--ip_address', type=str, dest='ip_address', help='mongodb ip address'
                , default =config.get('mongo', 'ip_address'))
    parser.add_argument('-p','--port', type=str, dest='port', help='mongodb port'
                , default =eval(config.get('mongo', 'port')))
    parser.add_argument('-dn','--db_name', type=str, dest='db_name', help='mongodb db name'
                , default =config.get('mongo', 'out_db'))
    parser.add_argument('-t','--tb_name', type=str, dest='tb_name', help='mongodb table name'
                , default =config.get('mongo', 'out_feat_tb'))
    # auth
    parser.add_argument('-un','--username', type=str, dest='username', help='mongodb username'
                , default =config.get('mongo', 'username'))
    parser.add_argument('-pw','--password', type=str, dest='password', help='mongodb password'
                , default =config.get('mongo', 'password'))  
                
    args = parser.parse_args()
    
    if args.folder:
        feat_dir = args.folder
    else:
        feat_dir  = config.get('app', 'HADOOP_MASTER')+'/user/hadoop/yigai/sality_virut_zbot_backdoor_hash_000'
    if args.name:
        file_name_given = args.name
    else:
        file_name_given  = 'aaaa'
    if args.out:
        out_dir = args.out
    else:
        out_dir  = 'out_result'
    if args.row_id:
        row_id_str = args.row_id
    else:
        row_id_str  = '88'

    if args.modelfolder:
        model_data_folder = args.modelfolder
    else:
        model_data_folder  = out_dir + '/' + row_id_str + '_model/'
    if args.listfile:
        list_file_test = args.listfile
    else:
        list_file_test  = out_dir + '/' + row_id_str + '_testhashlist.txt'
    if args.uploadtype:
        uploadtype = args.uploadtype
    else:
        uploadtype  = None
    if args.fromweb:
        fromweb = args.fromweb
    else:
        fromweb  = None
    if args.parameter:
        j_str = args.parameter
    else:
        j_str  = '{"learning_algorithm":"linear_svm_with_sgd", "c":"1", "iteration":"300", "regularization":"l2"}'
    if len(args.username)>0:
        username = args.username
    else:
        username  = None
    if len(args.password)>0:
        password = args.password
    else:
        password  = None 
    
    data_folder = feat_dir + "/"
    out_dir = out_dir + "/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
        
    if os.path.exists(model_data_folder):
        shutil.rmtree(model_data_folder)
    if not os.path.exists(model_data_folder):
        os.makedirs(model_data_folder)
    
    if os.path.isfile(list_file_test):
        try:
            os.remove(list_file_test)
        except OSError:
            pass
    else:
        with open(list_file_test, "w") as myfile:
            pass
    
    SparkContext.setSystemProperty('spark.rdd.compress', config.get('spark', 'spark_rdd_compress'))
    SparkContext.setSystemProperty('spark.driver.maxResultSize', config.get('spark', 'spark_driver_maxResultSize'))
    SparkContext.setSystemProperty('spark.executor.memory', args.exe_memory)
    SparkContext.setSystemProperty('spark.cores.max', args.core_max)

    sc = SparkContext(args.sp_master, 'sk-learn-train:'+str(args.row_id))
    
    t0 = time()
    
    
    ### load libsvm file ###
    libsvm_data_file = data_folder + "libsvm_data"
    print "libsvm_data_file:", libsvm_data_file
    samples_rdd = MLUtils.loadLibSVMFile(sc, libsvm_data_file)
    #print samples_rdd.count()
    labels_and_features_rdd = samples_rdd.map(lambda p: (p.label, p.features))
    all_data = labels_and_features_rdd.collect()
    features_list = [x.toArray() for _,x in all_data]
    labels_list = [x for x,_ in all_data]
    labels_list = array(labels_list)

    features_array = np.array(features_list)
    
    ### generate sparse matrix (csr) for all samples
    features_sparse_mtx = csr_matrix(features_array)
    
    ### randomly split the samples into training and testing data
    X_train_sparse, X_test_sparse, labels_train, labels_test = cross_validation.train_test_split(features_sparse_mtx, labels_list, test_size=0.4)

    t1 = time()
    print 'data generating time: %f' %(t1-t0)
    
    ###############################################
    ###########build learning model################
    ###############################################
    
    if flag_model == "linear_svm_with_sgd":
        ### 1: linearSVM
        print "====================1: Linear SVM============="
        clf = svm.LinearSVC()
        clf.fit(X_train_sparse, labels_train)
       
    #### save clf for future use ####
    joblib.dump(clf, model_data_folder + row_id_str+'.pkl') 
    
    #print "**model:coef***"
    #print clf.coef_
    #print "**model:intercept***"
    #print clf.intercept_
    
    ### Evaluating the model on testing data
    labels_pred = clf.predict(X_test_sparse)
    #print "************results*********"
    #print "Predicting results:"
    #print labels_pred
    #print "True testing labels:"
    #print labels_test
    
    accuracy = clf.score(X_test_sparse, labels_test)
    print "Accuracy = ", accuracy
    
    ###################################################
    ### generate label names (family names) ###########
    ### connect to database to get the column list which contains all column number of the corresponding feature####
    ###################################################
    key = "dic_name_label"
    jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
    jstr_proj='{"value":1}'
            
    doc=query_mongo.find_one(args.ip_address, args.port, args.db_name, args.tb_name, username, password, jstr_filter, jstr_proj)
    dic_list = doc['value']
    
    label_dic = {}
    for i in range(0, len(dic_list)):
        for key in dic_list[i]:
            label_dic[dic_list[i][key]] = key.encode('UTF8')
    print "label_dic:", label_dic
    
    labels_list = []
    for key in sorted(label_dic):
        labels_list.append(label_dic[key])
    
    ### generate sample numbers of each family in testing data###
    testing_sample_number = len(labels_test)
    print "testing_sample_number:", testing_sample_number
    test_cnt_dic = {}
    for key in label_dic:
        test_cnt_dic[key] = 0
    for i in range (0, testing_sample_number):
        for key in label_dic:
            if labels_test[i] == key:
                test_cnt_dic[key] = test_cnt_dic[key] + 1
    print "Number of samples in each label is:", test_cnt_dic
    
    ###############################################
    ###########plot prediction result figure#######
    ###############################################
    
    ### reorder labels so that labels are ordered according to the true label of the data
    len_pred = len(labels_pred)
    wide_len = math.ceil(math.sqrt(len_pred))
    
    pred_list = labels_pred.tolist()
    test_list = labels_test.tolist()
    labels_true_pred = zip(test_list, pred_list)
    labels_true_pred.sort(key=lambda x: x[0])
    
    test_ordered = [x for x,_ in labels_true_pred]
    pred_ordered = [x for _,x in labels_true_pred]
    
    
    last_value = test_ordered[len_pred - 1]
    for i in range(len_pred, int(wide_len*wide_len)):
        test_ordered.append(last_value)
        pred_ordered.append(last_value)
    
    mtx_testing = np.reshape(test_ordered, (wide_len, wide_len))
    mtx_pred = np.reshape(pred_ordered, (wide_len, wide_len))
    
        
    ### plot figues ###
    fig, ax = plt.subplots()
    cax = ax.imshow(mtx_pred, interpolation='nearest', cmap=plt.cm.jet)
    num_labels = len(labels_list)
    tic = range(0, num_labels)
    
    labels_str=[]
    # append sample count at the end
    for key in sorted(test_cnt_dic):
        labels_str.append(labels_list[key] + "("+str(test_cnt_dic[key])+")" )   
    
    cbar = fig.colorbar(cax, ticks=tic)
    cbar.ax.set_yticklabels(labels_str)# vertically oriented colorbar
    cbar.ax.invert_yaxis()
    plt.xlabel('Prediction (Single Run)')
    plt.savefig(out_dir+file_name_given+"_1"+".png")
    
    
    fig, ax = plt.subplots()
    cax = ax.imshow(mtx_testing, interpolation='nearest', cmap=plt.cm.jet)
    #ax.set_title('Gaussian noise with vertical colorbar')
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=tic)
    cbar.ax.set_yticklabels(labels_str)# vertically oriented colorbar
    cbar.ax.invert_yaxis()
    plt.xlabel('True Labels (Single Run)')
    plt.savefig(out_dir+file_name_given+"_2"+".png")

    plt.show()
    
    
    #############################################################
    ###################for 2 class only (plot ROC curve)#########
    #############################################################
    if len(labels_list) == 2:
        
        reverse_label_dic = dict((v,k) for k, v in label_dic.items())
        if 'clean' in reverse_label_dic:
            flag_clean = reverse_label_dic['clean']
        else:
            print "No ROC curve generated: 'clean' must be a label for indicating negative class!"
            return
            
        confidence_score = clf.decision_function(X_test_sparse)
                
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
        
        
        ###########plot ROC figure#######
        try:
            fpr, tpr, thresholds = roc_curve(s_labels, scores, pos_label = 1)
            roc_auc = auc(fpr, tpr)
        except ValueError as e:
            print "Error!! in ROC curve: ",
            print e
        
        print "ROC_AUC = ", roc_auc
        
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.savefig(out_dir+file_name_given+"_ROC"+".png")
        print "Figure save!"
        
        #### generate fpr tpr ACC threshold results file###
        ROC_file = out_dir+file_name_given+"_ROC_value.txt"
        if os.path.exists(ROC_file):
            try:
                os.remove(ROC_file)
            except OSError, e:
                print ("Error: %s - %s." % (e.ROC_file,e.strerror))
        
        for i in range(0, len(fpr)):
            ACC = (testing_P*tpr[i] + testing_N*(1-fpr[i]))/(testing_P + testing_N)
            with open(ROC_file, 'a') as f:
                f.write('%0.5f  ' % (fpr[i]))
                f.write('%0.5f  ' % (tpr[i]))
                f.write('%0.5f  ' % (thresholds[i]))
                f.write('%0.5f\n' % (ACC))
            
        ### print FPR, TPR, ACC ###
        if flag_model == "linear_svm_with_sgd":
            thr = 0
        elif flag_model == "logistic_regression_with_lbfgs" or flag_model == "logistic_regression_with_sgd":
            thr = -0.5
        for i in range(0, len(thresholds)):
            if thresholds[i] < thr:
                print "===Results Summary==="
                print "Accuracy: ", accuracy
                #print "Accuracy (calculate): ", (testing_P*tpr[i-1] + testing_N*(1-fpr[i-1]))/(testing_P + testing_N)
                print "False Positive Rate (FPR): ", fpr[i-1]
                print "True Positive Rate (TPR): ", tpr[i-1]
                print "====================="
                break

                
    # only update db for web request
    if fromweb=="1": 
        #print "database update"
        str_sql="UPDATE atdml_document set "+"accuracy = '"+str(accuracy*100)+"%" \
            +"', status = 'learned', processed_date ='"+str(datetime.datetime.now()) \
            +"' where id="+row_id_str
        ret=exec_sqlite.exec_sql(str_sql)
        print "Data update done! ret=", str(ret)
    else:
        print "accuracy = '"+str(accuracy*100)+"%"
    
    t1 = time()
    print 'running time: %f' %(t1-t0)
    
    print 'Finished!'
    return 0

if __name__ == '__main__':
    __description__ = "ML single run to show accuracy, roc, generate model, etc"
    main()
