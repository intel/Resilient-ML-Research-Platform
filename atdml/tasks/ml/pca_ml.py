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



####pyspark#####
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import PCA, PCAModel
from pyspark.ml.linalg import Vectors, SparseVector, DenseVector

#####matplotlib###############
import matplotlib
matplotlib.use('Agg')

#####import for django database####
sys.path.append('./db')
import exec_sqlite

####import our own library####
import query_mongo
import ml_util
from ml_util import *
import zip_feature_util
import pydoop.hdfs as hdfs


CONF_FILE='../../app.config' # at the base dir of the web
config=ConfigParser.ConfigParser()
config.read(CONF_FILE)

def main():
    
    parser = ArgumentParser(description=__description__)
    parser.add_argument("-f", "--folder", type=str, metavar="folder of features", help="hdfs folder contains features", required=False)
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
        pca_jstr  = '{"threshold":0.9,"lib":"mllib"}'

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
        model_data_folder  = None
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
        zip_file_name  = 'pca_ml.zip'

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
    , args.sp_master,config.get('spark', 'spark_rdd_compress'),config.get('spark', 'spark_driver_maxResultSize')
    , args.exe_memory, args.core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples, fromweb, pca_jstr
    , 'pca_ml:'+row_id_str, model_data_folder )
    
    
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
        lib='mllib'
  
    ret=-1
    # start here =================================================================== ===============
    t0 = time()
    
    # source libsvm filename  
    libsvm_data_file = os.path.join(hdfs_feat_dir , "libsvm_data")
    print "INFO: libsvm_data_file=", libsvm_data_file
    
    # load sample RDD from text file   
    # format Row(label, features, hash) from get_sample_dataframe() 
    samples_df, feature_count = zip_feature_util.get_sample_dataframe(sc, libsvm_data_file, 0, None)
    print "INFO: feature_count=",feature_count

    #df_pcaed format: hash,label, features 
    (df_pcaed, k, pca_model)=PCA_transform(sc, samples_df,feature_count, threshold, k) 
    print "INFO: Doing PCA... threshold=",threshold,",k=",k
    #print "df_pcaed=",df_pcaed.first()
    #print "k=",k
    #print "pca_model=",pca_model
    #print "pc=",pca_model.pc

    # pca model filename ============================= ===============
    if model_data_folder is None:
        if row_id_str != ds_id:
            # get from parent dataset
            model_data_folder  = os.path.join(config.get('app', 'HADOOP_MASTER'),config.get('app', 'HDFS_MODEL_DIR'), ds_id+"_pca")
        else:
            model_data_folder  = os.path.join(config.get('app', 'HADOOP_MASTER'),config.get('app', 'HDFS_MODEL_DIR'), row_id_str+"_pca")
            
    # create HDFS folder
    try:
        hdfs.mkdir(model_data_folder)
    except IOError as e:
        print "WARNING: I/O error({0}): {1}".format(e.errno, e.strerror),". At HDFS=", save_dir
    except:
        print "WARNING: Unexpected error:", sys.exc_info()[0] ,". At HDFS=", save_dir
    
        
    if not threshold is None:
        #pca_fname=os.path.join(hdfs_feat_dir , row_id_str+'_pca_'+str(threshold)+'.ml')
        pca_fname=os.path.join(model_data_folder , 'pca_model_'+str(threshold))
        libsvm_data_pca = os.path.join(hdfs_feat_dir , "libsvm_data_pca_"+str(threshold)+'.ml')
    else:
        pca_fname=os.path.join(model_data_folder , 'pca_model_'+str(k))
        libsvm_data_pca = os.path.join(hdfs_feat_dir , "libsvm_data_pca_"+str(k)+'.ml')
    
    # save pca model to HDFS ===============
    print "INFO: pca_fname=",pca_fname
    pca_model.write().overwrite().save(pca_fname)
    
    # save pca data to HDFS ============================= ===============
    print "INFO: libsvm_data_pca=",libsvm_data_pca
    # construct libsvm string
    libsvm_rdd=df_pcaed.rdd.map(lambda p: p[0]+" "+str(int(p[1]))+zip_feature_util.dv2libsvm(p[2].toArray()))
    
    # clean up old libsvm file ============================= ===============
    try:
        hdfs.rmr(libsvm_data_pca)
    except IOError as e:
        print "WARNING: I/O error({0}): {1}".format(e.errno, e.strerror)
    except:
        print "WARNING: Unexpected error at rmr():", sys.exc_info()[0]     

    # overwrite pca file at hdfs
    libsvm_rdd.saveAsTextFile(libsvm_data_pca)

    
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
        print "INFO: Update Sqlite DB done! ret=", str(ret)

    
    t1 = time()
    print 'INFO: running time: %f' %(t1-t0)
    
    #print 'Finished!'
    return 0

# PCA transform df by either k/n_components  or threshold ========================
def PCA_transform(sc, samples_df,feature_count, threshold, k):
    # check input
    if threshold and ((threshold > 1) or (threshold < 0)):
        print "ERROR: PCA_transform: Input threshold should be within 0 to 1"
        return (None,None,None)
    if k and k<0:
        print "ERROR: transform: Input k should be greater than 0"
        return (None,None,None)
    #print "df.shape=",df.shape
        
    #print "in ml_sklearn_PCA_transform()"
    df_reduced=None
    pca=None
    if not threshold is None: # by threshold ===============
        if feature_count> 200:
            fk=200
            print "INFO: force k to "+str(fk)+" for PCA."
        else:
            fk=feature_count
            
        pca = PCA(k=fk, inputCol="features", outputCol="pcaFeatures")
        pca_model=pca.fit(samples_df)
        sum_ratio = 0
        # get ratio array and find n_components 
        var_arr=pca_model.explainedVariance
        print "RESULT: PCA ratio_vec=",var_arr
        
        n_components=ml_util.ml_get_n_components(var_arr,threshold)
        '''
        for n_components,val in enumerate(var_arr):
            sum_ratio=sum_ratio+val
            if sum_ratio >= threshold:
                break
        '''
        k=n_components
        #print sum_ratio, n_components  

        df_pcaed_all = pca_model.transform(samples_df).select("hash","label","pcaFeatures")
        # get k column only
        sqlCtx = SQLContext(sc)
        df_pcaed=sqlCtx.createDataFrame(
            df_pcaed_all.rdd.map(lambda p: (p["hash"],p["label"],p["pcaFeatures"].toArray()[:k]) )
            .map(lambda p: Row(hash=p[0],label=p[1],pcaFeatures=DenseVector(p[2])))
        )
        print "INFO: PCA_transform: n_components =",n_components, ", threshold=",threshold
    elif k >0: # by n_components  ===============
        pca = PCA(k=k, inputCol="features", outputCol="pcaFeatures")
        pca_model=pca.fit(samples_df)
        df_pcaed = pca_model.transform(samples_df).select("hash","label","pcaFeatures")
        print "INFO: PCA_transform: n_components =",k
    
    return (df_pcaed, k, pca_model)   


    
if __name__ == '__main__':
    __description__ = "ML single run to show accuracy, roc, generate model, etc"
    main()
