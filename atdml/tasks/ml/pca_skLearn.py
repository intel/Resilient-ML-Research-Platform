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

from argparse import ArgumentParser
from scipy.sparse import *
from scipy import *
from time import time
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from scipy.stats import entropy
from sets import Set
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix


####pyspark#####
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.util import MLUtils

#####matplotlib###############
import matplotlib
matplotlib.use('Agg')
'''
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.colors as colors
import matplotlib.cm as cmx
'''
#####import for django database####
sys.path.append('./db')
import exec_sqlite

####import our own library####
import query_mongo
import ml_util
from ml_util import *
import zip_feature_util


CONF_FILE='../../app.config' # at the base dir of the web
config=ConfigParser.ConfigParser()
config.read(CONF_FILE)

def main():
    
    parser = ArgumentParser(description=__description__)
    parser.add_argument("-f", "--folder", type=str, metavar="folder of features", help="folder contains features, hdfs://xxx.com:9000/user/...", required=False)
    parser.add_argument("-n", "--name", type=str, metavar="file name", help="file name for sample folder", required=False)
    parser.add_argument("-o", "--out", type=str, metavar="out figure folder", help="folder contains output", required=False)
    parser.add_argument("-r", "--row_id", type=str, metavar="row id", help="row_id number in the db", required=False)
    parser.add_argument("-mf", "--modelfolder", type=str, metavar="model folder", help="model for prediction", required=False)
    parser.add_argument("-w", "--fromweb", type=str, metavar="flag for web", help="flag for web", required=False)
    parser.add_argument("-pm", "--parameter", type=str, metavar="parameters in json", help="json string contains learning alg and parameter selection", required=False)
    #parser.add_argument("-sl", "--showlabelname", type=str, metavar="show label name", help="0: not shown; 1: show label name", required=False)
    parser.add_argument("-dsid", "--ds_id", type=str, metavar="source dataset id", help="source dataset id for training option", required=False)

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
    # zip spark code for workers
    parser.add_argument("-zo", "--zipout_dir", type=str, metavar="out figure folder", help="folder contains python code", required=False)
    parser.add_argument("-zc", "--zipcode_dir", type=str, metavar="out code", help="out code folder for python code", required=False)
    parser.add_argument("-zf", "--zipfilename", type=str, metavar="python code zip file", help="python code zip file for distribution to works", required=False)
                
    args = parser.parse_args()
    # hdfs out dir
    if args.folder:
        feat_dir = args.folder
    else:
        feat_dir  = config.get('app', 'HADOOP_MASTER')+'/user/hadoop/data_out/245/'
    if args.name:
        file_name_given = args.name
    else:
        file_name_given  = 'libsvm_data'
    # local_out_dir at web local (not including dataset id)
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
        pca_jstr = args.parameter
    else:
        pca_jstr  = '{"k":2,"threshold":0.9,"lib":"sklearn"}'

    if len(args.username)>0:
        username = args.username
    else:
        username  = None
    if len(args.password)>0:
        password = args.password
    else:
        password  = None 
    # model will be web local
    if args.modelfolder:
        model_data_folder = args.modelfolder
    else:
        model_data_folder  = os.path.join(local_out_dir,row_id_str , row_id_str + '_model')
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
        zip_file_name  = 'pca_scikit.zip'

    # hdfs dir for final data output    
    hdfs_feat_dir = feat_dir 
    
    if row_id_str != str(ds_id):
        local_out_dir=os.path.join(local_out_dir,ds_id)
    else: 
        local_out_dir=os.path.join(local_out_dir,row_id_str)
    
    # create local folder for output
    if not os.path.exists(local_out_dir):
        os.makedirs(local_out_dir)

    # mongo info for connection
    mongo_tuples=(args.ip_address, args.port, args.db_name, args.tb_name, username, password)

    return pca(row_id_str, ds_id, hdfs_feat_dir, local_out_dir  
    , args.sp_master,config.get('spark', 'spark_rdd_compress'),config.get('spark', 'spark_driver_maxResultSize'), args.exe_memory, args.core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples, fromweb, pca_jstr
    , 'pca_sklearn:'+row_id_str, model_data_folder )
    
    
# ================================================================================== train () ============ 
def pca(row_id_str, ds_id, hdfs_feat_dir, local_out_dir  
    , sp_master, spark_rdd_compress, spark_driver_maxResultSize, sp_exe_memory, sp_core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples, fromweb, pca_jstr
    , jobname, model_data_folder ): 
    
    # create zip files for Spark workers ================= ================
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
    
    '''
    SparkContext.setSystemProperty('spark.rdd.compress', config.get('spark', 'spark_rdd_compress'))
    SparkContext.setSystemProperty('spark.driver.maxResultSize', config.get('spark', 'spark_driver_maxResultSize'))
    SparkContext.setSystemProperty('spark.executor.memory', args.exe_memory)
    SparkContext.setSystemProperty('spark.cores.max', args.core_max)

    sc = SparkContext(args.sp_master, 'pca_sklearn:'+str(args.row_id))
    '''
    
    pca_param=json.loads(pca_jstr)
    if "k" in pca_param:
        k=pca_param["k"]
    else:
        k=None
    if "threshold" in pca_param:
        threshold=pca_param["threshold"]
    else:
        threshold=None         
    if "lib" in pca_param:
        lib=pca_param["lib"]
    else:
        lib='sklearn'
    #if "recreate" in pca_param:
    #    recreate=pca_param["recreate"]
    #else:
    #    recreate='0'
        
    ret=-1
    # start here =================================================================== ===============
    t0 = time()
    
    # source libsvm filename  
    libsvm_data_file = os.path.join(hdfs_feat_dir , "libsvm_data")
    print "INFO: libsvm_data_file=", libsvm_data_file

    # load feature count file
    feat_count_file=libsvm_data_file+"_feat_count"
    feature_count=zip_feature_util.get_feature_count(sc,feat_count_file)
    print "INFO: feature_count=",feature_count
    
    #samples_rdd = MLUtils.loadLibSVMFile(sc, libsvm_data_file)
    
    # load sample RDD from text file   
    # format (LabeledPoint,hash) from str2LabeledPoint_hash() 
    samples_rdd, feature_count = zip_feature_util.get_sample_rdd(sc, libsvm_data_file, feature_count, '')
    
    # convert labeled point to tuple (label,features)
    #labels_and_features_rdd = samples_rdd.map(lambda p: (p.label, p.features))
    
    all_data = samples_rdd.collect()
    sample_count=len(all_data)

    # 2-D array
    features_list = [x.features.toArray() for x,_ in all_data]
    # label array
    labels_list_all = [x.label for x,_ in all_data]
    # hash array
    hash_list_all = [x for _,x in all_data]
    
    # convert to int to speed up 
    features_array = np.array(features_list, dtype = np.int32)
    true_label_array = np.array(labels_list_all, dtype = np.int8)
    hash_list_all=np.array(hash_list_all)

    # PCA here, TBD for removal
    print "INFO: features_array.shape=",features_array.shape
    print "INFO: true_label_array.shape=",true_label_array.shape
    print "INFO: hash_list_all.shape=",hash_list_all.shape

    print "INFO: Doing PCA... threshold=",threshold,",k=",k
    (features_array_reduced, k, pca) = sklearn_PCA_transform(features_array, threshold, k)
    #print "features_array_reduced t=", type(features_array_reduced), ",features_array_reduced=", features_array_reduced
    print "INFO: features_array_reduced.shape", features_array_reduced.shape,",k=",k

    pca_fname3=""
    if not threshold is None:
        pca_fname=os.path.join(model_data_folder , row_id_str+'_pca_'+str(threshold)+'.pkl')
        pca_fname3=os.path.join(model_data_folder , row_id_str+'_pca3_'+str(threshold)+'.pkl')
    else:
        pca_fname=os.path.join(model_data_folder , row_id_str+'_pca_'+str(k)+'.pkl')
    
    #pc_3=features_array_reduced[:,0:3]
    #print "type=",type(pc_3),"shape=",pc_3.shape,"pca_fname3=",pca_fname3
    #np.save(pca_fname3,pc_3)
    
    if not os.path.exists(model_data_folder):
        os.makedirs(model_data_folder)
    if os.path.exists(pca_fname): # remove file if exists
        try:
            for fl in glob.glob(pca_fname+"*"):
                os.remove(fl)
        except OSError, e:
            print ("Error: %s - %s." % (e.pca_fname,e.strerror))
    print "INFO: pca model file at",pca_fname

    # save pca into a file
    joblib.dump(pca, pca_fname) 
    
    
    # create local libsvm file ===========================
    if not features_array_reduced is None:
        # output to local temp file 
        pca_libsvm_filename=os.path.join(local_out_dir, "libsvm_data_pca_"+str(k))
        print "INFO: target local pca_libsvm_filename=",pca_libsvm_filename 
        f=ml_recreate_file_4write(pca_libsvm_filename)
        if not f is None:
            for idx,arr in enumerate(features_array_reduced):
                line=str(hash_list_all.item(idx))+" "+str(true_label_array.item(idx))
                for idx, val in enumerate(arr):
                    line=line+" "+str(idx+1)+":"+str(val)
                line=line+"\n"
                #print "line=",line 
                f.write(line)
            f.close()
    
    # dest hdfs filename
    #libsvm_data_pca = os.path.join(hdfs_feat_dir , "libsvm_data_pca_"+str(k))
    # sklearn use threashold as filename
    libsvm_data_pca = os.path.join(hdfs_feat_dir , "libsvm_data_pca_"+str(threshold))
    print "INFO: target HDFS libsvm_data_pca=",libsvm_data_pca
    
    # overwrite pca file at hdfs
    ret=ml_overwrite_hdfs_file(pca_libsvm_filename, libsvm_data_pca)
    '''
    # local .pickle file
    reduced_feat_filename=os.path.join(local_out_dir,row_id_str+"_feat_pca_"+str(threshold)+".pickle")
    print "reduced_feat_filename=",reduced_feat_filename
    # if not exist, create one
    if not os.path.exists(reduced_feat_filename) or recreate=='1':
        # check if re-create:

        print "Doing PCA..., shape=", features_array.shape
        (features_array_reduced, k) = sklearn_PCA_transform(features_array, threshold, k)
        print "*** k=",k
        #print "Done PCA"
        if not features_array_reduced is None:
            ml_pickle_save(features_array_reduced, reduced_feat_filename)
        else: 
            return -1 # error occur
        #print "features_array_reduced created!"
    else:
        features_array_reduced=ml_pickle_load(reduced_feat_filename)  
        print "features_array_reduced loaded!"
    '''

    
    t1 = time()
    print 'INFO: PCA processing time: %f' %(t1-t0)
    
    ### insert pca_param into mongoDB  ###
    filter='{"rid":'+row_id_str+',"key":"pca_param"}'
    if not threshold is None:
        pca_param["threshold"]=threshold
    if not k is None:
        pca_param["k"]=k
    print "INFO: pca_param=",pca_param
    upsert_flag=True
    jstr_insert = '{ "rid":'+row_id_str+',"key":"pca_param", "value":'+json.dumps(pca_param)+'}'
    ret=query_mongo.upsert_doc_t(mongo_tuples,filter,jstr_insert,upsert_flag)
    print "INFO: Upsert count for pca_param=",ret
    
    # only update db for web request   ===========
    if fromweb=="1": 
        #print "database update"
        str_sql="UPDATE atdml_document set "  \
            +" status = 'pca-ed', processed_date ='"+str(datetime.datetime.now()) \
            +"' , ml_pca_opts = '"+json.dumps(pca_param) \
            +"' where id="+row_id_str
        ret=exec_sqlite.exec_sql(str_sql)
        if ret==1:
            print "INFO: PCA done! SQL update ret=", str(ret)

    
    t1 = time()
    print 'INFO: running time: %f' %(t1-t0)
    
    #print 'Finished!'
    return 0

# PCA transform X by either k/n_components  or threshold ========================
def sklearn_PCA_transform(X, threshold, k):
    # check input
    if threshold and ((threshold > 1) or (threshold < 0)):
        print "ERROR: sklearn_PCA_transform: Input threshold should be within 0 to 1"
        return (None,None)
    if k and k<=0:
        print "ERROR: sklearn_PCA_transform: Input k should be greater than 2"
        return (None,None)
    #print "X.shape=",X.shape
        
    #print "in ml_sklearn_PCA_transform()"
    X_reduced=None
    pca=None
    if not threshold is None: # by threshold ===============
        #pca = PCA(n_components=X.shape[1])
        pca = PCA(n_components=threshold) #, svd_solver ='full' 
        pca.fit(X)
        ratio_vec = pca.explained_variance_ratio_
        sum_ratio = 0
        for i in range(len(ratio_vec)):
            sum_ratio = sum_ratio + ratio_vec[i]
            if sum_ratio >= threshold:
                n_components = i+1 
                break
        X_tr = pca.transform(X)
        print "INFO: X_tr.shape=",X_tr.shape
        
        if len(X_tr[0])<3:
            print "ERROR: Dataset has dimension less than 3"
            
        elif n_components <3:
            print "WARNING: set n_components to 3 for 3D graph"
            n_components=3
        
        X_reduced = X_tr[:,0:n_components]
        k=n_components
        print "INFO: sklearn_PCA_transform: n_components =",n_components, ", threshold=",threshold
        print "RESULT: PCA ratio_vec=",ratio_vec
    elif k >0: # by n_components  ===============
        pca = PCA(n_components=k)    
        pca.fit(X)
        X_reduced = pca.transform(X)
        print "INFO: PCA n_components =",k
    
    if not X_reduced is None:
        print "INFO: sklearn_PCA_transform: X_reduced.shape=",X_reduced.shape

    return (X_reduced, k, pca)   
    


    
if __name__ == '__main__':
    __description__ = "ML single run to show accuracy, roc, generate model, etc"
    main()
