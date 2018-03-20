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


# our own library imports
sys.path.append('./ml')
from zip_preprocess_pattern import preprocess_pattern, convert_to_line, convert_to_line_by_bash
from zip_feature_extraction_ngram import feature_extraction_ngram, tokenize_by_dict, dict2nparr

#####import for django database####
sys.path.append('./db')
import query_mongo
import exec_sqlite
import ml_util

CONF_FILE='../../app.config' # at the base dir of the web
config=ConfigParser.ConfigParser()
config.read(CONF_FILE)
KERAS_LIB_DIR=config.get('env', 'KERAS_LIB_DIR')
sys.path.append(KERAS_LIB_DIR)
from keras.models import load_model
BATCH_SIZE=128

def main():

    parser = ArgumentParser(description=__description__)
    parser.add_argument("-d", "--name", type=str, metavar="file name", help="file name for prediction", required=False)
    parser.add_argument("-o", "--out", type=str, metavar="learner output", help="out files for prediction", required=False)
    
    parser.add_argument("-r", "--row_id", type=str, metavar="row_id number", help="row_id number in the db", required=False)
    #parser.add_argument("-ng", "--num_gram", type=str, dest='num_gram', help="num gram for svm"
    #    , default =config.get('machine_learning', 'svm_num_gram'))

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

    parser.add_argument("-mfn", "--model_filename", type=str, metavar="filename of model in json"
        , help="json filename; expect coef_arr and coef_intercept", required=False)
    parser.add_argument("-mj", "--model_json", type=str, metavar="ML model in json string"
        , help="ML model in json string", required=False)
    parser.add_argument("-sj", "--sample_json", type=str, metavar="sample in json string"
        , help="sample in json string", required=False)
    parser.add_argument("-pfn", "--pca_filename", type=str, metavar="filename of PCA model in pickle"
        , help="pickle filename for PCA model", required=False)

    # data format
    parser.add_argument("-mc", "--metadata_count", type=str, metavar="metadata fields in raw data", help="metadata fields in raw data", required=False)
    parser.add_argument("-di", "--data_idx", type=str, metavar="array index for log data", help="array index for log data", required=False)
    parser.add_argument("-li", "--label_idx", type=str, metavar="array index for label", help="array index for label", required=False)
    parser.add_argument("-ptn", "--pattern_str", type=str, metavar="regular express pattern to extract string"
        , help="regular express pattern to extract string", required=False)
    parser.add_argument("-ld", "--ln_delimitor", type=str, metavar="delimiter to separate log string into lines", help="delimiter to separate log string into lines", required=False)
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
        local_out_dir = args.out
    else:
        local_out_dir  = '.'
    if args.row_id:
        row_id_str = args.row_id
    else:
        row_id_str  = '553'
    if args.ds_id:
        ds_id = args.ds_id
    else:
        ds_id  = row_id_str 
    if args.cid:
        cid_str = args.cid
    else:
        cid_str  = '01'

    ###################################################
    if args.num:
        num_gram = eval(args.num)
    else:
        num_gram  = eval(config.get("machine_learning","svm_num_gram"))

    if args.max:
        MAX_FEATURES = eval(args.max)
    else:
        MAX_FEATURES  = eval(config.get("machine_learning","MAX_FEATURES"))
    if args.fromweb:
        fromweb = args.fromweb
    else:
        fromweb  = None        

    if args.metadata_count:
        metadata_count = eval(args.metadata_count)
    else:
        metadata_count  = 3 # label,md5,date,filetype
    if args.data_idx:
        data_idx = eval(args.data_idx)
    else:
        data_idx  = 3 # label,md5,date,filetype, log_data_starthere; 
    if args.label_idx:
        label_idx = eval(args.label_idx)
    else:
        label_idx  = 0 # label,md5,date,filetype, log_data_starthere; 
    if args.pattern_str:
        pattern_str = args.pattern_str
    else:
        pattern_str  = r'^I/AndroidATD\([ ]*[\d]+\): \[API\] (.*) \[.*' # 
    if args.ln_delimitor:
        ln_delimitor = args.ln_delimitor
    else:
        ln_delimitor  = '\t'

    if args.parameter:
        j_str = args.parameter
    else:
        j_str='{"c":"1","iterations":"300","regularization":"l2","learning_algorithm":"logistic_regression_with_sgd"}'
    if args.lib: # mllib or scikit
        lib_mode = args.lib
    else:
        lib_mode='scikit'
    if args.showlabelname: # mllib or scikit
        labelnameflag = eval(args.showlabelname)
    else:
        labelnameflag = 1
    if args.verbose: 
        verbose = args.verbose
    else:
        verbose = "1"

    if args.model_filename and len(args.model_filename)>0:
        model_filename = args.model_filename
    else:
        model_filename  = None
    if args.pca_param:
        pca_param = json.loads(args.pca_param)
    else:
        pca_param  = None
    if args.pca_filename:
        pca_filename = args.pca_filename
    else:
        pca_filename  = None
    if args.model_json:
        model_json = args.model_json
    else:
        model_json  = None
    if args.sample_json:
        sample_json = args.sample_json
    else:
        sample_json  = None        
 
    ######database########################################
    if len(args.username)>0:
        username = args.username
    else:
        username  = None
    if len(args.password)>0:
        password = args.password
    else:
        password  = None     
    
    #
    binary_flag=True # TBD for param

    
    return predict(row_id_str, ds_id, cid_str, input_gz, local_out_dir, num_gram
        , j_str, lib_mode
        , fromweb, verbose,label_idx, data_idx, metadata_count, pattern_str, ln_delimitor, binary_flag, labelnameflag
        , model_filename, model_json, sample_json, pca_filename, pca_param
        , args.sp_master, args.exe_memory, args.core_max, MAX_FEATURES , dic_name_label=None
        , feat_cnt_threshold=args.feat_cnt_threshold
        , ip_address=args.ip_address, port=args.port, db_name=args.db_name, tb_name=args.tb_name
        , username=username, password=password
        )
        
    '''
    ret=predict_single_file_pattern.predict(
     row_id_str="314729",ds_id="314729",cid_str="315273"
    ,input_gz="/home/django/myml/media/result/314729/an_00acd0d9c55559d0a26469b67a23c2142b59e18f208109a20dfec68c1e756c37.only.log.dirty.gz"
    ,local_out_dir="/home/django/myml/media/result/314729"
    ,num_gram=2,j_str='{"c":"1","learning_algorithm":"linear_svm"}'
    ,lib_mode='scikit',fromweb='2'
    ,verbose=0,label_idx=0,data_idx=3,metadata_count=3,pattern_str='(.*)')    
    '''
    
#  Used by massive prediction too ================================= =======================
def predict(row_id_str, ds_id, cid_str, input_gz, local_out_dir, num_gram
        , j_str, lib_mode
        , fromweb, verbose,label_idx=0, data_idx=3, metadata_count=3, pattern_str='(.*)', ln_delimitor = '\t', binary_flag=True, labelnameflag=1
        , model_filename=None, str_model_json=None, sample_txt=None ,pca_filename=None, pca_param=None
        , sp_master=config.get('spark', 'spark_master'), exe_memory=config.get('spark', 'spark_executor_memory')
        , core_max=config.get('spark', 'spark_cores_max')
        , MAX_FEATURES=eval(config.get("machine_learning","MAX_FEATURES")) , dic_name_label=None
        , feat_cnt_threshold=config.get('machine_learning', 'feature_count_threshold')
        , ip_address=config.get('mongo', 'out_ip_address'), port=eval(config.get('mongo', 'out_port'))
        , db_name=config.get('mongo', 'out_db'), tb_name=config.get('mongo', 'out_tb')
        , username=config.get('mongo', 'out_username'), password=config.get('mongo', 'out_password')
        , sc=None
        ):

    print "in"
    t0 = time()
    coef_arr=None
    dic_hash_str=None
    dic_seq_hashes=None
    dic_hashes_seq=None
    feat_sample_count_arr=None
    hashes_cnt_dic=None
    hash_str_dic=None
    data_rows=None
    data_cols=None

    ml_opts=None

    # load model from strings ============ for offline IN, liner model ==============
    if not str_model_json is None and len(str_model_json)>10:
        try:
            model_json=json.loads(str_model_json)
        except Exception as e:
            print "ERROR: model json load error." , e
            return -1
      
        #print "model_json=",model_json
        if "coef_arr" in model_json:
            coef_arr=model_json["coef_arr"]
            col_num = len(coef_arr)   
        if "coef_intercept" in model_json:
            coef_intercept=model_json["coef_intercept"]
        if "dic_hash_str" in model_json:
            dic_hash_str=model_json["dic_hash_str"]
        if "dic_seq_hashes" in model_json:
            dic_seq_hashes=model_json["dic_seq_hashes"]    
        if "pca_param" in model_json:
            pca_param=model_json["pca_param"]
        if "feat_sample_count_arr" in model_json:
            feat_sample_count_arr=model_json["feat_sample_count_arr"]
        if dic_name_label is None:
            dic_name_label=model_json["dic_name_label"]
        if j_str is None:
            j_str=model_json["ml_opts"]
        ml_opts=json.loads(j_str)
        #print "j_str=",j_str
        num_gram=eval(model_json["ml_n_gram"])
        lib_mode="offline"
    # load model from a file ======
    elif not model_filename is None: 
        if os.path.exists(model_filename):
            print "INFO: model from file=",model_filename
            try:
                with open(model_filename) as jf:
                    model_json=json.load(jf)
                    if "coef_arr" in model_json:
                        coef_arr=model_json["coef_arr"]
                        col_num = len(coef_arr)   
                    if "coef_intercept" in model_json:
                        coef_intercept=model_json["coef_intercept"]
                    if "dic_hash_str" in model_json:
                        dic_hash_str=model_json["dic_hash_str"]
                    if "dic_seq_hashes" in model_json:
                        dic_seq_hashes=model_json["dic_seq_hashes"]    
                    if "pca_param" in model_json:
                        pca_param=model_json["pca_param"]
                    if "feat_sample_count_arr" in model_json:
                        feat_sample_count_arr=model_json["feat_sample_count_arr"]
                    dic_name_label=model_json["dic_name_label"]
                    ml_opts=model_json["ml_opts"]
                    num_gram=eval(model_json["ml_n_gram"])
                    lib_mode="offline"
                    #print "-- id=",id,",ds_id=",ds_id
                    print "INFO: model feature count=",col_num
                    print "INFO: model for num_gram=",num_gram
                    #print "model json=",model_json
            except Exception as e:
                print "ERROR: loading model file ["+model_filename+"] error! ",e
                return -3
        else:
            print "ERROR: model file ["+model_filename+"] not found! "
            return -4
            
    # ML parameters ===================== input ml_opts ==============
    learning_algorithm=None
    try:
        if ml_opts is None:
            ml_opts = json.loads(j_str)
        #print "INFO: ml_opts=",ml_opts
        if 'learning_algorithm' in ml_opts:
            learning_algorithm = ml_opts['learning_algorithm'] 
    except Exception as e:
        print "WARNING: load learning_algorithm failed.",e
    print "INFO: learning_algorithm=",learning_algorithm
    
    
    # read raw data from .gz file ===================== input .gz ==============
    #if json_str is None:
    # TBD by faster convert_to_line_by_bash()
    try:
        f = gzip.open(input_gz, 'rb')
        sample_txt = convert_to_line(f, metadata_count) # check if one line, if raw file then convert to 1 line
        #print "sample_txt=",sample_txt[:100].replace('\t',',')
        #print "sample_txt=",sample_txt.replace('\t',',')
        f.close()
    except Exception as e:
        print "ERROR: load data file ["+input_gz+"] failed.",e
        return -5

    #f = gzip.open(input_gz, 'rb')
    #file_content = f.readline() # assume only one line
    #file_content=convert_to_line(f, metadata_count) # check if one line, if raw file then convert to 1 line
    #f.close()

    # input: assume one line of ngram pattern format string ===========
    #       return an array [meta-data1,meta-data2,...,str_arr]
    raw_arr=None
    coef_arr=None
    feat_arr=None

    # input:  one line text
    #       return array: [meta-data1,meta-data2,..., hash_cnt_dic, hash_str_dic]
    raw_arr=preprocess_pattern(sample_txt, metadata_count, pattern_str, ln_delimitor, label_idx, label_arr=None )
    #print "*****************raw_arr=",raw_arr
    
    # input:  array: [meta-data1,meta-data2,..., hash_cnt_dic, hash_str_dic]
    #       return hashes_cnt_dic: {hash,hash:count),...}  hash_str_dic: {hash: 'str1',... }
    feat_arr=feature_extraction_ngram(raw_arr, data_idx, MAX_FEATURES, num_gram)
    #print "**************feat_arr=",feat_arr

    #
    if feat_arr is None or len(feat_arr)==0:
        print "ERROR: Raw data format error or no feature found at predict_single_file_pattern."
        return -1

    # load PCA params ================= ========
    threshold=None
    n_component=None
    # data for PCA; TBD for all algorithm
    if learning_algorithm =='kmeans' : #
        if pca_param is None:
            # get from mongo 
            key = "pca_param"
            jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
            jstr_proj='{"value":1}'  

            # ???get parent dataset's data
            #if ds_id != row_id_str:
            #    jstr_filter='{"rid":'+ds_id+',"key":"'+key+'"}'
                    
            doc=query_mongo.find_one(ip_address, port, db_name, tb_name, username, password, jstr_filter, jstr_proj)
            if doc and "value" in doc:
                pca_param = doc['value']
                print "INFO: pca_param=", pca_param

        # param for PCA model and transform: expect both threshold and k in pca_param
        if not pca_param is None:
            if "threshold" in pca_param:
                threshold=pca_param["threshold"]
            if "k" in pca_param:
                n_component=pca_param["k"]
        print "INFO: n_component=",n_component,", threshold=",threshold        
        
    # get {seq :hash,hash } mapping from mongo  key=dic_seq_hashes ===================
    if dic_seq_hashes is None:
        key = "dic_seq_hashes"
        jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
        jstr_proj='{"value":1}'
        #print "************** ds_id=",ds_id,", rid=",row_id_str
        # get parent dataset's data
        if ds_id != row_id_str:
            jstr_filter='{"rid":'+ds_id+',"key":"'+key+'"}'
        
        doc=query_mongo.find_one(ip_address, port, db_name, tb_name, username, password, jstr_filter, jstr_proj)
        if not doc is None:
            dic_seq_hashes = doc['value']
        else:
            # get from local file
            fn=os.path.join(local_out_dir,ds_id+"_dic_seq_hashes.pkl")
            print "INFO: get dic_seq_hashes from local", fn
            dic_seq_hashes=ml_util.ml_pickle_load(fn)
            print "INFO: len(dic_seq_hashes)=", len(dic_seq_hashes)
        
    if dic_seq_hashes:
        dic_len=len(dic_seq_hashes)
    else:
        dic_len=0

    # print feature for ref by a new optional param?
    out_f=None
    # for feature list. not for kmeans 
    if verbose=="1" and learning_algorithm not in ('kmeans'):
        # get {hash : raw string} mapping ==================================
        if dic_hash_str is None:
            key = "dic_hash_str"  #{"123":"openFile"}
            jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
            jstr_proj='{"value":1}'
            # get parent dataset's data
            if ds_id != row_id_str:
                jstr_filter='{"rid":'+ds_id+',"key":"'+key+'"}'
                
            doc=query_mongo.find_one(ip_address, port, db_name, tb_name, username, password, jstr_filter, jstr_proj)
            if not doc is None:
                dic_hash_str = doc['value']
            else:
                # get from local file
                fn=os.path.join(local_out_dir,ds_id+"_dic_hash_str.pkl")
                print "INFO: get dic_hash_str from local", fn
                dic_hash_str=ml_util.ml_pickle_load(fn)
                print "INFO: len(dic_hash_str)=", len(dic_hash_str)
        
            #print "hashes_cnt_dic=",hashes_cnt_dic
            #print "dic_hash_str=",dic_hash_str
        
        # clean up feature file
        out_file=os.path.join(local_out_dir,cid_str+"_feature_list.json")
        if os.path.exists(out_file):
            try:
                os.remove(out_file)
            except OSError, e:
                print ("ERROR: %s - %s." % (e.strerror, out_file))
        if not learning_algorithm in ('kmeans','lstm','cnn'):
            print "INFO: feature file=",out_file
            out_f=open(out_file, 'a')
        
        # get local dict    
        hash_str_dic=feat_arr[data_idx+1]
        # convert key to string
        hash_str_dic={str(k): v for k, v in hash_str_dic.items()}

        coef_arr=None
        # get coef_arr ==================================
        if coef_arr is None and not learning_algorithm in ('kmeans','lstm','cnn'):
            key = "coef_arr"  #{"123":"openFile"}
            jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
            jstr_proj='{"value":1}'
            # each model has its own coef_arr
                
            doc=query_mongo.find_one(ip_address, port, db_name, tb_name, username, password, jstr_filter, jstr_proj)
            if not doc is None and 'value' in doc:
                coef_arr = doc['value']
            else:
                # get from local file
                fn=os.path.join(local_out_dir,ds_id+"_coef_arr.pkl")
                print "INFO: get coef_arr from local fn=", fn
                coef_arr=ml_util.ml_pickle_load(fn)
                print "INFO: len(coef_arr)=", len(coef_arr)

        # get feat_sample_count_arr ==================================
        if feat_sample_count_arr is None and not learning_algorithm in ('kmeans'):
            key = "feat_sample_count_arr"  
            jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
            jstr_proj='{"value":1}'
            # get parent dataset's data
            if ds_id != row_id_str:
                jstr_filter='{"rid":'+ds_id+',"key":"'+key+'"}'
                
            doc=query_mongo.find_one(ip_address, port, db_name, tb_name, username, password, jstr_filter, jstr_proj)
            if not doc is None:
                feat_sample_count_arr = doc['value']
            else:
                # get from local file
                fn=os.path.join(local_out_dir,ds_id+"_feat_sample_count_arr.pkl")
                print "INFO: get feat_sample_count_arr from local", fn
                feat_sample_count_arr=ml_util.ml_pickle_load(fn)
                print "INFO: len(feat_sample_count_arr)=", len(feat_sample_count_arr)  
                 
    # reverse key/value
    if dic_seq_hashes:
        dic_hashes_seq={v: k for k, v in dic_seq_hashes.items()}
        #print "dic_hashes_seq=",len(dic_hashes_seq)
    
    # get hashes_cnt_dic
    if feat_arr and feat_arr[data_idx]:
        hashes_cnt_dic=feat_arr[data_idx]
    #if hashes_cnt_dic:
    #    print "hashes_cnt_dic len=",len(hashes_cnt_dic)
    #print "hash_str_dic=",hash_str_dic
    
    curr_dic = {}
    fout_arr=[]
    if not feat_sample_count_arr is None and len(feat_sample_count_arr)>0:
        has_fcount=True
    else:
        has_fcount=False
        
    # convert to sequential feature number by {dic_hashes_seq} & write feat_list file
    for hashes in hashes_cnt_dic:
        description_str=""
        feat_out={}
        if hashes in dic_hashes_seq:
            if binary_flag:
                curr_dic[dic_hashes_seq[hashes] ]=1  #force binary 
            else:
                curr_dic[dic_hashes_seq[hashes] ]=hashes_cnt_dic[hashes] 
                
            if verbose=="1" and not learning_algorithm in ('kmeans'):
                description_str = ml_util.feats2strs(hashes,dic_hash_str)
                feat_out["fid"]=int(dic_hashes_seq[hashes])
                feat_out["ngram"]=hashes
                feat_out["desc"]=description_str
                if not dic_hashes_seq is None and not dic_hashes_seq[hashes] is None and not coef_arr is None:
                    feat_out["coef"]=coef_arr[int(dic_hashes_seq[hashes])-1]
                else:
                    feat_out["coef"]=None
                if has_fcount:
                    feat_out["feat_sample_count"]=feat_sample_count_arr[int(dic_hashes_seq[hashes])-1]              
                else:
                    feat_out["feat_sample_count"]="n/a"
                fout_arr.append(feat_out)
                #out_f.write('%s\t%s\t%s\n' % (dic_hashes_seq[hashes],hashes,description_str))
                
                print "INFO: f=",int(dic_hashes_seq[hashes])-1,hashes,description_str
        else:
            if verbose=="1" and not learning_algorithm in ('kmeans','lstm'):
                # not in dataset; use local dict
                description_str = ml_util.feats2strs(hashes,hash_str_dic)
                feat_out["fid"]="None"
                feat_out["ngram"]=hashes
                feat_out["desc"]=description_str
                feat_out["coef"]=0
                # TBD here
                feat_out["feat_sample_count"]="n/a"
                fout_arr.append(feat_out)
                #out_f.write('%s\t%s\t%s\n' % ("None",hashes,description_str))
                #print "    Non-existing f=",description_str
                print "INFO: Feature '"+hashes+ "' not found,",description_str
        
        
    s_feat_count=len(curr_dic) 
    print "INFO: sample feature count=",s_feat_count
    
    # save feature list file ================================== ============
    if verbose=="1" and not out_f is None:
        if len(fout_arr) > 0:
            out_f.write(json.dumps(fout_arr))               
        out_f.close()
        print "INFO: End Feature list ===================================="
    
    predict_val=None
    sing_label_pred=None
    
    
    #print "lib_mode=",lib_mode
    # fit model here =============================================== FIT ===============================
    if lib_mode == "scikit": #"SKlean":
        print "INFO: Predict by sklearn library ***"
        # get the ML model
        model_file  = os.path.join(local_out_dir , row_id_str + '_model/' + row_id_str + '.pkl')

        # load clf from model file
        sk_model = joblib.load(model_file)
        clf_coef_arr=None
        intercept_arr=None
        
        print "INFO: clf=",sk_model
        #print "sk_model __class__=",sk_model.__class__," __name__=",sk_model.__class__.__name__

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
            print "WARNING: Can't get sk_model.coef_[0]. e=",e
            col_num = dic_len #how to get feature number for sparse array? 
        print "INFO: total feature # in sklearn model: ", col_num

        
        # generate the sparse matrix from curr_dic 
        sparse_test = ml_util.generate_matrix_from_dic(curr_dic, col_num)
        
        # Predict here ======================================== .predict() =====================
        # May need PCA transfor here; kmeans has PCA by default
        if learning_algorithm =='kmeans' : #
            print "INFO: in scikit kmeans"
            #pca_filename=os.path.join(local_out_dir , row_id_str + '_model' , row_id_str+'_pca_0.9' + '.pkl')
            if pca_filename is None:
                if threshold is None:
                    print "ERROR: threshold for Sklearn PCA is required!"
                    return -8
                pca_filename=os.path.join(local_out_dir , row_id_str + '_model' , row_id_str+'_pca_'+str(threshold) + '.pkl') 
            print "INFO: pca_filename=", pca_filename
            if n_component is None:
                print "ERROR: n_component for PCA is required!"
                return -9
                
            # load PCA model ============
            pca = joblib.load(pca_filename)
            # convert to dense array for transform
            tr_x= pca.transform(sparse_test.toarray())
            #n_component=pca.n_components
            labels_pred=sk_model.predict([tr_x[0][:n_component]])
        else:    
            labels_pred = sk_model.predict(sparse_test)
            
        sing_label_pred = labels_pred[0]
        #print "--type(labels_pred)=",type(labels_pred),",len=",len(labels_pred.tolist()),",labels_pred="
        #for i in labels_pred.tolist():
        #    print i
        
        # calculate hypothesis value ================
        if not clf_coef_arr is None and not intercept_arr is None:
            clf_classname=sk_model.__class__.__name__
            predict_val=ml_util.calculate_hypothesis(curr_dic, col_num, clf_coef_arr, intercept_arr[0], clf_classname)
            print "INFO: intercept_arr[0]=",intercept_arr[0], ", len=", len(intercept_arr)
            #??print "--threshold=",sk_model.threshold
            print "INFO: clf_coef_arr size=",clf_coef_arr.size
            print "INFO: clf_classname=",clf_classname,", h(wx)=",predict_val
    
    elif lib_mode == "mllib": # spark mllib
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
        
        if sc is None:
            sc=get_sc(row_id_str,sp_master, exe_memory, core_max)

            
            '''
            SparkContext.setSystemProperty('spark.rdd.compress', config.get('spark', 'spark_rdd_compress'))
            SparkContext.setSystemProperty('spark.driver.maxResultSize', config.get('spark', 'spark_driver_maxResultSize'))
            SparkContext.setSystemProperty('spark.executor.memory', exe_memory)
            SparkContext.setSystemProperty('spark.cores.max', core_max)

            sc = SparkContext(sp_master, 'single_predict:'+row_id_str)
            '''
            
        model_name = learning_algorithm       
        save_dir = config.get('app', 'HADOOP_MASTER')+config.get('app', 'HDFS_MODEL_DIR')+'/'+row_id_str

        print "before load model **************** model dir=",save_dir
        
        if model_name == "linear_svm_with_sgd":
            mllib_model = SVMModel.load(sc, save_dir)
            col_num = len(mllib_model.weights)
        elif model_name == "logistic_regression_with_lbfgs" or model_name == "logistic_regression_with_sgd":
            mllib_model = LogisticRegressionModel.load(sc, save_dir)
            col_num = mllib_model.numFeatures # len(mllib_model.weights) return 3x value
        elif model_name == "kmeans":
            mllib_model = KMeansModel.load(sc, save_dir)
            col_num =len(mllib_model.clusterCenters[0])
        else:
            print "Training model selection error: no valid ML model selected!"
            return
        # get the model dimension
        #col_num = len(mllib_model.weights)
        print "INFO: total feature # in mllib model: ", col_num
        
        # calculate hypothesis value ================
        model_weight=None
        if learning_algorithm not in ("kmeans") :
            model_weight=mllib_model.weights
            intercept=mllib_model.intercept 
            
        coef_arr=None
        predict_val=None    
        if not model_weight is None :
            coef_arr=np.asarray(model_weight.toArray().tolist())
        #print "model_weight=" ,model_weight  
        #print "type(model_weight)=" ,type(model_weight)  #pyspark.mllib.linalg.DenseVector
        #print "type(coef_arr)=" ,type(coef_arr)  #'numpy.ndarray'
        #print "type(intercept)=" ,type(intercept)       #float        
        print "INFO: coef_arr shape=",coef_arr.shape

        # pca transform curr_dic if needed:
        #   pca for clustering: load transformed libsvm data back to curr_dic 
        if learning_algorithm in ("kmeans") :
            libsvm_str=ml_util.dict2libsvm_str("0", curr_dic, set_all_one="Y")
            #print "INFO: PCA transform for kmeans; libsvm_str=",libsvm_str
            pca_k=n_component  # mllib use n_component as model filename
            new_libsvm_str=ml_util.ml_mllib_pca_local_transform(row_id_str,libsvm_str,pca_k,ds_id)
            (lbl, curr_dic)=ml_util.libsvm_str2tuple(new_libsvm_str)
            print "INFO: PCA transformed. curr_dic len=", len(curr_dic)
        
        
        if not coef_arr is None and not intercept is None:
            classname=mllib_model.__class__.__name__
            predict_val=ml_util.calculate_hypothesis(curr_dic, col_num, coef_arr, intercept, classname)
            print "INFO: mllib_model.intercept=",intercept
            print "INFO: mllib_model.threshold=",mllib_model.threshold
            print "INFO: coef_arr size=",coef_arr.size
            print "INFO: classname=",classname,", h(wx)=",predict_val
   
        # generate the sparse matrix from curr_dic
        vector_test = ml_util.generate_vector_from_dic(curr_dic, col_num)
        # use API to generate vector; tbd not set to 1 for values
        #vector_test=SparseVector(col_num,curr_dic)
        #print "vector_test=",vector_test
        
        sing_label_pred = mllib_model.predict(vector_test)         
        
    elif lib_mode == "dnn": # DNN ==============================
        # TBD, need to do CNN here (feed forward data)
        print "INFO: learning_algorithm=",learning_algorithm
        if learning_algorithm == "cnn":
            # generate the sparse matrix from curr_dic 
            #print "curr_dic=",curr_dic
            X_pred = np.array([dict2nparr(curr_dic, dic_len)])
            
            # reshape for image data
            if data_rows is None and not ml_opts is None:
                if "data_rows" in ml_opts:
                    data_rows=int(ml_opts["data_rows"])
                if "data_cols" in ml_opts:
                    data_cols=int(ml_opts["data_cols"])
            if not data_rows is None and not data_cols is None and data_rows>0 and data_cols>0:
                X_pred = X_pred.reshape(X_pred.shape[0], data_rows, data_cols,1)
            print "INFO: X_pred shape=", X_pred.shape # , type(X_pred)


        elif learning_algorithm == "lstm":
            # below is for LSTM (recurrent)
            # str_hash_dict: str to hash mapping
            str_hash_dict={v: k for k, v in dic_hash_str.iteritems()}
            # build token_dict: str 2 seq mapping
            token_dict={}
            for k,v in str_hash_dict.iteritems():
                token_dict[k]=int(dic_hashes_seq[str(v)])
            
            #print "raw_arr=",raw_arr
            #print "token_dict=",token_dict
            #print "data_idx=",data_idx,",label_idx=",label_idx
            arr=tokenize_by_dict(raw_arr, data_idx, token_dict,label_idx, label_dict=None)
            #print "arr=",arr
            
            # convert data in arr[data_idx] to np array
            X_pred=np.array([arr[data_idx]])
        
        
        
        # load DNN model
        # get the ML model
        model_file  = os.path.join(local_out_dir , row_id_str + '_model', row_id_str + '_model.h5')
        print "INFO: model_file=", row_id_str + '_model.h5'
        model = load_model(model_file)

        # predict
        pred_arr=model.predict(X_pred, batch_size=BATCH_SIZE, verbose=0)
        #print "pred_arr=",pred_arr
        # convert to list
        pred_list=pred_arr.tolist()
        sing_label_pred=0
        predict_val = max(pred_list[0])
        #print "predict_val=",predict_val
        sing_label_pred=pred_list[0].index(predict_val)
        if len(pred_list[0])==2 and sing_label_pred==0:
            predict_val = 1-predict_val # for clean/0, reverse the value
        print "RESULT: predict score=",predict_val
        
        

    elif lib_mode == "offline": # predict offline for classification only ==============================
        #print "INFO: Offline prediction ===="
        #classname=ml_opts["learning_algorithm"]
        predict_val=None
        # no predict value for clustering
        if learning_algorithm not in ('kmeans'):
            predict_val=ml_util.calculate_hypothesis(curr_dic, col_num, coef_arr, coef_intercept, learning_algorithm)
        sing_label_pred=0
        if learning_algorithm and "logistic" in learning_algorithm.lower():
            if predict_val>=0.5:
                sing_label_pred=1
        else:
            if predict_val>=0:
                sing_label_pred=1
        #print "INFO: algorithm=",learning_algorithm  #,", h(wx)=",predict_val
        print "RESULT: predict score=",predict_val

    print "RESULT: predict output=", sing_label_pred      

    ### generate label names (family names) #####
    ### connect to database to get the column list which contains all column number of the corresponding feature####
    pred_label=None
    label_dic = {}
    if labelnameflag == 1:
        if dic_name_label is None:
            key = "dic_name_label"
            jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
            jstr_proj='{"value":1}'

            # get parent dataset's data
            if ds_id != row_id_str:
                jstr_filter='{"rid":'+ds_id+',"key":"'+key+'"}'
            
            doc=query_mongo.find_one(ip_address, port, db_name, tb_name, username, password, jstr_filter, jstr_proj)
            dic_list = doc['value']
            
        for i in range(0, len(dic_list)):
            for key in dic_list[i]:
                label_dic[dic_list[i][key]] = key.encode('UTF8')
        print "INFO: label_dic=", label_dic
        

        if not sing_label_pred is None:
            pred_label = label_dic[int(sing_label_pred)]
        if learning_algorithm in ("kmeans"):
            pred_label = "cluster# "+str(sing_label_pred)
        print "RESULT: prediction=", pred_label
    
    status="predicted"
    print "INFO: feat_cnt_threshold=",feat_cnt_threshold
    # check if over threshold
    if not feat_cnt_threshold is None and s_feat_count < int(feat_cnt_threshold) and not learning_algorithm in "lstm":
        print "WARNING: feature count for this sample='"+str(s_feat_count)+"' is less than threshold '"+str(feat_cnt_threshold)+"'"
        pred_label="not_enough_info"
        status="new"
    ###################################
    ############update DB##############
    ###################################
    
    # only update db for web request
    if fromweb=="1": 
        #print "database update"
        str_sql="UPDATE atdml_document set status = '"+status+"', processed_date ='" \
            +str(datetime.datetime.now())+"', prediction = '"+ str(pred_label)  \
            +"', predict_val = '"+str(predict_val) \
            +"' where id="+cid_str
        ret=exec_sqlite.exec_sql(str_sql)
        #print "Data update done! ret=", str(ret)
    elif fromweb=="2":
        return {"id":int(cid_str),"opt_id":int(row_id_str),"ds_id": int(ds_id),"prediction":str(pred_label),"predict_val":predict_val,"learning_algorithm":learning_algorithm \
            ,"lib":lib_mode, "ml_opts":ml_opts,"predict_index": int(sing_label_pred) }
        #print "prediction: '"+ pred_label+"'"
    
    
    t1 = time()
    print 'INFO: total running time: %f' %(t1-t0)
    return 0

#  return spark context
def get_sc(row_id_str,sp_master, exe_memory, core_max):
    
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
    SparkContext.setSystemProperty('spark.executor.memory', exe_memory)
    SparkContext.setSystemProperty('spark.cores.max', core_max)

    sc = SparkContext(sp_master, 'single_predict:'+row_id_str)
    
    return sc
    
# process raw input to 
#   array: [meta-data1,meta-data2,..., hash_cnt_dic, hash_str_dic]
#       hashes_cnt_dic: {hash,hash:count),...}  hash_str_dic: {hash: 'str1',... }
def preprocess_one_line(one_line, num_gram, metadata_count=3, pattern_str='(.*)', ln_delimitor='\t' \
    ,label_idx=0, data_idx=3, label_arr=None):


    # featuring data here; assume one line of ngram pattern format string ===========
    #       an array [meta-data1,meta-data2,...,str_arr]
    raw_arr=preprocess_pattern(one_line, metadata_count, pattern_str, ln_delimitor, label_idx, label_arr )
    #print "*****************raw_arr len=",len(raw_arr[metadata_count])
    #t1 = time()
    #print 'INFO: after preprocess_pattern psfp: %f' %(t1-t0)
    
    #   array: [meta-data1,meta-data2,..., hash_cnt_dic, hash_str_dic]
    #       hashes_cnt_dic: {hash,hash:count),...}  hash_str_dic: {}
    feat_arr=feature_extraction_ngram(raw_arr, data_idx, num_gram=num_gram \
        , MAX_FEATURES=eval(config.get("machine_learning","MAX_FEATURES")) , has_str_dic='N' )
    #print "**************feat_arr len =",len(feat_arr[metadata_count])
    #t1 = time()
    #print 'INFO: after feature_extraction_ngram psfp: %f' %(t1-t0)
    
    return feat_arr
    


if __name__ == '__main__':
    __description__ = "single file prediction for pattern"
    main()
