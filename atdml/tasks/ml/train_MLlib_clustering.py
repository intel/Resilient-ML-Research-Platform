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
from sklearn.metrics import roc_curve, auc, adjusted_mutual_info_score, adjusted_rand_score

####pyspark#####
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.linalg import Vectors 
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.clustering import KMeans, KMeansModel, GaussianMixture, GaussianMixtureModel
from pyspark.ml.feature import PCA as PCAml
from pyspark.mllib.feature import PCA as PCAmllib



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
portion = eval(config.get("machine_learning","training_portion"))
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

# find the max index number in the libsvm list: "index:value"
def find_max(arr):
    max=0
    for i in arr:
        n=i.split(':')
        if len(n)>1 and int(n[0])>max:
            max=int(n[0])
    return max

# convert string "index:value" to json {index:int}
def to_label_int_json_tuple(arr):
    ret_arr={}
    label=arr[0]
    for i in arr[1]:
        n=i.split(':')
        if len(n)>1:
            ret_arr[int(n[0])]=int(n[1])
    return (label, ret_arr )    
    
def main():
    
    parser = ArgumentParser(description=__description__)
    parser.add_argument("-f", "--folder", type=str, metavar="folder of features", help="hdfs folder contains features", required=False)
    parser.add_argument("-n", "--name", type=str, metavar="file name", help="filename for hdfs input data", required=False)
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
    # zip spark code
    parser.add_argument("-zo", "--zipout_dir", type=str, metavar="out figure folder", help="folder contains python code", required=False)
    parser.add_argument("-zc", "--zipcode_dir", type=str, metavar="out code", help="out code folder for python code", required=False)
    parser.add_argument("-zf", "--zipfilename", type=str, metavar="python code zip file", help="python code zip file for distribution to works", required=False)
                
    
    args = parser.parse_args()
    
    if args.folder:
        hdfs_feat_dir = args.folder
    else:
        hdfs_feat_dir  = config.get('app', 'HADOOP_MASTER')+'/user/hadoop/yigai/sality_virut_zbot_backdoor_hash_000'
    if args.name:
        src_filename = args.name
    else:
        src_filename  = 'aaaa'
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
        zip_file_name  = 'train_ml_clustering.zip'

    # mongo info for connection
    mongo_tuples=(args.ip_address, args.port, args.db_name, args.tb_name, username, password)

    return train(row_id_str, ds_id, hdfs_feat_dir, local_out_dir, ml_opts_jstr
    , args.sp_master,config.get('spark', 'spark_rdd_compress'),config.get('spark', 'spark_driver_maxResultSize'), args.exe_memory, args.core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples, labelnameflag, fromweb, src_filename
    , 'train_ml_clustering:'+row_id_str )

# ================================================================================== train () ============ 
def train(row_id_str, ds_id, hdfs_feat_dir, local_out_dir, ml_opts_jstr
    , sp_master, spark_rdd_compress, spark_driver_maxResultSize, sp_exe_memory, sp_core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples, labelnameflag, fromweb, src_filename
    , jobname ): 

    # create zip files for Spark workers ================= ================
    zip_file_path = ml_build_zip_file(zipout_dir, zipcode_dir, zip_file_name, prefix='zip_feature_util')
    print "INFO: zip_file_path=",zip_file_path
        
    #data_folder = hdfs_feat_dir + "/"
    #local_out_dir = local_out_dir + "/"
    #if os.path.exists(local_out_dir): 
    #    shutil.rmtree(local_out_dir) # to keep smaplelist file
    if not os.path.exists(local_out_dir):
        os.makedirs(local_out_dir)
            
    # init Spark context ====
    sc=ml_util.ml_get_spark_context(sp_master
        , spark_rdd_compress
        , spark_driver_maxResultSize
        , sp_exe_memory
        , sp_core_max
        , jobname
        , [zip_file_path]) 

    # start here =================================================================== ===============
    t0 = time()
        
    
    ### Need to check if PCA available here ===========================
    libsvm_data_file = os.path.join(hdfs_feat_dir , src_filename) # need to set k numb in filename somehow
    print "INFO: libsvm_data_file=", libsvm_data_file
    #samples_rdd = MLUtils.loadLibSVMFile(sc, libsvm_data_file).cache()
    # load sample RDD from text file   
    # format (LabeledPoint,hash) from str2LabeledPoint_hash() 
    feature_count=0
    samples_rdd, feature_count = zip_feature_util.get_sample_rdd(sc, libsvm_data_file, feature_count, '')
    
    # get label as a list
    labels_list_all = samples_rdd.map(lambda p: int(p[0].label)).collect()
    total_sample_count=len(labels_list_all)
    parsedData =samples_rdd.map(lambda p: p[0].features).cache()
    #for i in parsedData.collect(): #p.features: pyspark.mllib.linalg.SparseVector
    #    print "pd=",type(i),",i=",i

    t1 = time()
    print 'INFO: running time: %f' %(t1-t0)
    t0 = t1
    
    ###############################################
    ########## build learning model ###############
    ###############################################
    
    ### get the parameters###
    print "INFO: ============Learning Algorithm and Parameters============="
    para_dict = json.loads(ml_opts_jstr)
    flag_model = para_dict['learning_algorithm'] # kmeans
    iteration_num = eval(para_dict['iterations'])
    k=2
    if 'k' in para_dict:
        k = eval(para_dict['k'])

    print "INFO: Learning Algorithm:", flag_model
    print "INFO: iterations=", iteration_num
    #print "training_sample_number=", training_sample_number
    
    ### generate label names (family names) #####
    ### connect to database to get the column list which contains all column number of the corresponding feature####
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
        print "INFO: generated label_dic:", label_dic 
        
    labels_list = []
    for key in sorted(label_dic):
        labels_list.append(label_dic[key])
    print "INFO: labels_list=", labels_list
    
    ### build model ###
    
    if flag_model == "kmeans":
        print "=================== Kmeans ============"
        model = KMeans.train(parsedData, k, maxIterations=iteration_num)   
        t_cost= model.computeCost(parsedData)
        print "INFO: cost for training set =", str(t_cost)
        clusterCenters=model.clusterCenters
        print "INFO: clusterCenters t=", type(clusterCenters)  #list
    elif flag_model == "gaussian_mixture_model": # didn't work some native lib issue
        print "=================== Gaussian_Mixture_Model ============"
        model = GaussianMixture.train(parsedData, k, maxIterations=iteration_num)   
        print "INFO: model.weights =", model.weights
    else:
        print "INFO: Training model selection error: no valid ML model selected!"
        return
        
    ### Save model
    save_dir = config.get('app', 'HADOOP_MASTER')+config.get('app', 'HDFS_MODEL_DIR')+'/'+row_id_str
    try:
        hdfs.ls(save_dir)
        #print "find hdfs folder"
        hdfs.rmr(save_dir)
        #print "all files removed"
    except IOError as e:
        print "ERROR: I/O error({0}): {1}".format(e.errno, e.strerror)
    except:
        print "ERROR: Unexpected error:", sys.exc_info()[0] 
    
    print "INFO: model saved at hdfs=",save_dir
    print "INFO: model type=",type(model)," model=",model
    model.save(sc, save_dir)
        
    ###load model if needed 
    #sameModel = SVMModel.load(sc, save_dir)

    ### 
    # (true label, keams label, features list, hash)
    all_data=samples_rdd.map(lambda t: ( t[0].label, model.predict(t[0].features), t[0].features, t[1] ) ).collect() 
    true_label_arr = np.asarray([int(x) for x,_,_,_ in all_data])
    labels_kmeans = np.asarray([int(x) for _,x,_,_ in all_data])
    hash_list = np.asarray([x for _,_,_,x in all_data])
    print "INFO: all_data len=",len(all_data),"all_data t=",type(labels_list_all)
    print "INFO: true_label_arr.shape=",true_label_arr.shape,"labels_kmeans.shape=",labels_kmeans.shape
    print "INFO: true_label_arr t=",type(true_label_arr),"labels_kmeans t=",type(labels_kmeans)
    mtx_center=np.asarray(clusterCenters)
    features_array_reduced=np.asarray([x.toArray() for _,_,x,_ in all_data])
    print "INFO: mtx_center t=",type(mtx_center),"mtx_center.shape=",mtx_center.shape
    print "INFO: features_array_reduced t=",type(features_array_reduced),"features_array_reduced.shape",features_array_reduced.shape

    #Adjusted Mutual Information between two clusterings
    amis=adjusted_mutual_info_score(labels_list_all,labels_kmeans)
    print "INFO: Adjusted_mutual_info_score=", amis  
    #Similarity measure between two clusterings
    ars=adjusted_rand_score(labels_list_all,labels_kmeans)
    print "INFO: Adjusted_rand_score=", ars   

    
    accuracy=0.0
   
    t1 = time()
    print 'INFO: training run time: %f' %(t1-t0)
    t0 = t1

    ###############################################
    ########## plot histogram               ######
    ###############################################
    n_clusters=k
    plot_col_num = int(math.ceil(math.sqrt(n_clusters)))
    figsize = (4*plot_col_num, 3*int(math.ceil(n_clusters*1.0/plot_col_num)))
    

    print "INFO: n_clusters=",n_clusters,",label_dic=",label_dic
    print "INFO: plot_col_num=",plot_col_num,",figsize=",figsize,",local_out_dir=",local_out_dir
    
    # kmeans histogram
    _, p_true = ml_plot_kmeans_histogram_subfigures(true_label_arr, labels_kmeans, n_clusters, names = label_dic
                        , plot_col_num = plot_col_num, figsize=figsize, folder = local_out_dir, rid=row_id_str)
    # normalized kmeans histogram
    _, p_true_norm = ml_plot_kmeans_histogram_subfigures(true_label_arr, labels_kmeans, n_clusters, names = label_dic
                        , plot_col_num = plot_col_num, figsize=figsize, normalize = True, folder = local_out_dir, rid=row_id_str)
    

    ####plot "reverse" histogram with labels ####
    num_bars = max(true_label_arr) + 1
    figsize = (4*plot_col_num, 3*int(math.ceil(num_bars*1.0/plot_col_num)))
    
    _, p_cluster = ml_plot_kmeans_histogram_subfigures(labels_kmeans, true_label_arr, num_bars, names = label_dic
                        , plot_col_num = plot_col_num, figsize=figsize, reverse = True, folder = local_out_dir, rid=row_id_str)


    #### plot dot figures ####
    # dot plot for Kmeans   ===========
    filename=os.path.join(local_out_dir ,row_id_str+'_cluster.png')   
    filename_3d=os.path.join(local_out_dir ,row_id_str+'_cluster_3d.json')  
    ml_plot_kmeans_dot_graph_save_file(features_array_reduced, labels_kmeans, mtx_center, n_clusters, figsize=(10,7), filename=filename
        , title='KMeans', filename_3d=filename_3d)
        
    # dot plot for True Labels  ===========
    filename=os.path.join(local_out_dir ,row_id_str+'_cluster_tl.png')      
    filename_3d=os.path.join(local_out_dir ,row_id_str+'_cluster_3d_tl.json')  
    ml_plot_kmeans_dot_graph_save_file(features_array_reduced, true_label_arr, mtx_center, n_clusters, figsize=(10,7), filename=filename
        , title='True Labels', filename_3d=filename_3d)

    dataset_info={"training_fraction":1, "class_count":n_clusters,"dataset_count":total_sample_count}
    
    # only update db for web request
    if fromweb=="1": 
        #print "database update"
        str_sql="UPDATE atdml_document set "+"accuracy = '" \
            +"', status = 'learned', processed_date ='"+str(datetime.datetime.now()) \
            +"', total_feature_numb='"+str(feature_count) \
            +"', perf_measures='{}" \
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
