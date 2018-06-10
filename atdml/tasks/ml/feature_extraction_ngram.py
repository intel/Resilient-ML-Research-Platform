#! /usr/bin/python
'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''

# standard library imports
from argparse import ArgumentParser
import calendar
import itertools
import md5
import os
import re
import sys, ConfigParser
import socket
import zipfile
import shutil
from time import time
import collections, glob

# third party library imports
from pyspark import SparkContext
from pyspark import StorageLevel
from pyspark.sql import SQLContext
import ujson, json
from pprint import pprint as pp
from bson import json_util
from operator import add

# our own library imports
sys.path.append('./ml')
from zip_preprocess_pattern import preprocess_pattern, preprocess_json
from zip_feature_extraction_ngram import feature_extraction_ngram, tokenize_by_dict, djb2
#####import for mongodb ####
sys.path.append('./db')
import query_mongo
import exec_sqlite
import ml_util, fnmatch
import pydoop.hdfs as hdfs
from importlib import import_module
from sets import Set

CUSTOM_PREFIX='cf_'
CUSTOM_FUNC='featuring'
MAX_FILTER_LOWER_CNT=1000
metadata = 'metadata'

#########################
CONF_FILE='../../app.config' # at the base dir of the web
config=ConfigParser.ConfigParser()
config.read(CONF_FILE)
MAX_FEATURES  = eval(config.get("machine_learning","MAX_FEATURES"))
libsvm_alldata_filename = config.get("machine_learning","libsvm_alldata_filename")
dnn_alldata_filename = config.get("machine_learning","dnn_alldata_filename")

def arg_parser(parser):
    # input output file info
    parser.add_argument("-f", "--folder", type=str, metavar="folder of features", help="hdfs folder contains features", required=False)
    parser.add_argument('-d', '--dir_list', type = str, metavar = 'data root', help = 'hdfs dir for input', required =False)
    parser.add_argument("-o", "--local_out_dir", type=str, metavar="web local output folder for result", help="web local output folder for result", required=False)
    parser.add_argument("-mf", "--model_data_folder", type=str, metavar="model folder", help="model for prediction", required=False)
    parser.add_argument("-td", "--token_dict", type=str, metavar="dict for tokenize", help="convert string to number", required=False)

    #spark code info
    parser.add_argument("-zo", "--zipout_dir", type=str, metavar="out figure folder", help="folder contains python code", required=False)
    parser.add_argument("-zc", "--zipcode_dir", type=str, metavar="out code", help="out code folder for python code", required=False)
    parser.add_argument("-zf", "--zipfilename", type=str, metavar="python code zip file", help="python code zip file for distribution to works", required=False)
    
    #dataset info
    parser.add_argument("-r", "--row_id", type=str, metavar="row id", help="row_id number in the db", required=False)
    parser.add_argument("-ng", "--num_gram", type=str, dest='num_gram', help="num gram for svm"
        , default =config.get('machine_learning', 'svm_num_gram'))
    parser.add_argument("-w", "--fromweb", type=str, metavar="flag for web", help="flag for web", required=False)
    parser.add_argument("-ft", "--feature_count_threshold", type=str, dest='feature_count_threshold', help="filter sample with less than feature count"
        , default =config.get('machine_learning', 'feature_count_threshold'))
    parser.add_argument("-fr", "--filter_ratio", type=str, metavar="the ratio to exclude most top and bottom features by frequency"
        , help="the ratio to exclude most top and bottom features by frequency, should be a float >0 & <1", required=False)
    
    # data format
    parser.add_argument("-mc", "--metadata_count", type=str, metavar="metadata fields in raw data", help="metadata fields in raw data", required=False)
    parser.add_argument("-di", "--data_idx", type=str, metavar="array index for log data", help="array index for log data", required=False)
    parser.add_argument("-li", "--label_idx", type=str, metavar="array index for label", help="array index for label", required=False)
    parser.add_argument("-ptn", "--pattern_str", type=str, metavar="regular express pattern to extract string"
        , help="regular express pattern to extract string", required=False)
    parser.add_argument("-ld", "--ln_delimitor", type=str, metavar="delimiter to separate log string into lines", help="delimiter to separate log string into lines", required=False)
    parser.add_argument("-lba", "--label_arr", type=str, metavar="string array for label", help="string array for label; to verify data too"
        , default =None, required=False)
    parser.add_argument("-dfl", "--data_field_list", type=str, metavar="key name for json data"
        , help="key name for json data", required=False)
    parser.add_argument("-jkd", "--jkey_dict", type=str, metavar="dict for json data"
        , help="dict for json data", required=False)
        
    parser.add_argument("-cf", "--cust_featuring", type=str, metavar="custom featuring"
        , help="custom featuring", required=False)
    parser.add_argument("-cfp", "--cust_featuring_params", type=str, metavar="parameters for custom featuring"
        , help="parameters for custom featuring", required=False)
    parser.add_argument("-cfd", "--cust_folder", type=str, metavar="folder for user custom code of featuring"
        , help="folder for user custom code of featuring", required=False)

    # for output to MongoDB
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

    # spark            
    parser.add_argument('-sp','--sp_master', type=str, dest='sp_master', help='spark.master', default =config.get('spark', 'spark_master'))
    parser.add_argument('-em','--exe_memory', type=str, dest='exe_memory', help='spark.executor.memory' \
        , default =config.get('spark', 'spark_executor_memory'))
        #, default ='12g')
    parser.add_argument('-cm','--core_max', type=str, dest='core_max', help='spark.cores.max' \
        , default =config.get('spark', 'spark_cores_max'))

    return parser.parse_args()

def main():  # ============= =============  ============= =============
    # parse arguments
    parser = ArgumentParser(description=__description__)
    args = arg_parser(parser)

    if args.dir_list:
        hdfs_dir_list = args.dir_list
    else:
        hdfs_dir_list  = config.get('app', 'HADOOP_MASTER')+'/user/hadoop/cyang8/data_pdf/'
    if args.local_out_dir:
        local_out_dir = args.local_out_dir
    else:
        local_out_dir  = '/home/django/myml/media/result'
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
        zip_file_name  = 'feature_ngram.zip'
    if args.cust_folder:
        cust_folder = args.cust_folder
    else:
        cust_folder  = './user_custom'
          
    if args.row_id:
        row_id_str = str(args.row_id)
    else:
        row_id_str  = ""
    if len(args.username)>0:
        username = args.username
    else:
        username  = None
    if len(args.password)>0:
        password = args.password
    else:
        password  = None

    if args.num_gram:
        num_gram = eval(args.num_gram)
    else:
        num_gram = 2
    if args.fromweb:
        fromweb = args.fromweb
    else:
        fromweb  = None
        
    if args.metadata_count:
        metadata_count = eval(args.metadata_count)
    else:
        metadata_count  = 3 # label,md5,date
    if args.data_idx:
        data_idx = eval(args.data_idx)
    else:
        data_idx  = 3 # label,md5,date, log_data_starthere; 
    if args.label_idx:
        label_idx = eval(args.label_idx)
    else:
        label_idx  = 0 # label,md5,date,filetype, log_data_starthere; 
    if args.pattern_str:
        pattern_str = args.pattern_str
    else:
        pattern_str  = r'(.*) ' # 
    if args.ln_delimitor:
        ln_delimitor = args.ln_delimitor
    else:
        ln_delimitor  = '\t' # 
    if args.folder:
        hdfs_feat_dir = args.folder
    else:
        hdfs_feat_dir  = config.get('app', 'HADOOP_MASTER')+config.get('app','FEATURE_DES_DIR')+'/'+row_id_str
    if args.label_arr and len(args.label_arr)>0:
        try: # convert to array
            label_arr = eval(args.label_arr)
        except:
            print "ERROR: Error in label_arr=",args.label_arr
            label_arr= None #['clean','dirty']
    else:
        label_arr  = None # ['clean','dirty']
    # for json data    
    if args.data_field_list:
        data_field_list = args.data_field_list
    else:
        data_field_list  = None
    if args.jkey_dict:
        jkey_dict = args.jkey_dict
    else:
        jkey_dict  = '{"meta_list":["label","md5","mdate"], "data_key":"logs"}'
    # model will be web local
    if args.model_data_folder:
        model_data_folder = args.model_data_folder
    else:
        model_data_folder  = os.path.join(local_out_dir,row_id_str , row_id_str + '_model')
    if args.token_dict:
        token_dict = json.loads(args.token_dict)
    else:
        token_dict  = None
    if args.cust_featuring:
        cust_featuring = args.cust_featuring
    else:
        cust_featuring  = None
    if args.cust_featuring_params:
        cust_featuring_params = args.cust_featuring_params
    else:
        cust_featuring_params  = None
    if args.filter_ratio:
        filter_ratio = eval(args.filter_ratio)
    else:
        filter_ratio  = None
    # mongo info for connection
    mongo_tuples=(args.ip_address, args.port, args.db_name, args.tb_name, username, password)

    return feat_extr_ngram(row_id_str, hdfs_dir_list, hdfs_feat_dir, model_data_folder
    , args.sp_master,config.get('spark', 'spark_rdd_compress'),config.get('spark', 'spark_driver_maxResultSize'), args.exe_memory, args.core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples, fromweb, label_arr, metadata_count,label_idx,data_idx, pattern_str, ln_delimitor, data_field_list, jkey_dict
    , 'feature_pattern_ngram:'+row_id_str, num_gram, args.feature_count_threshold ,token_dict
    , config.get('env', 'HDFS_RETR_DIR')
    , cust_featuring=cust_featuring, cust_featuring_params=cust_featuring_params, local_out_dir=local_out_dir
    , filter_ratio=filter_ratio
    )
    
    
# ================================================================================== train () ============ 
def feat_extr_ngram(row_id_str, hdfs_dir_list, hdfs_feat_dir, model_data_folder
    , sp_master, spark_rdd_compress, spark_driver_maxResultSize, sp_exe_memory, sp_core_max
    , zipout_dir, zipcode_dir, zip_file_name 
    , mongo_tuples, fromweb, label_arr, metadata_count,label_idx,data_idx, pattern_str, ln_delimitor, data_field_list, jkey_dict
    , jobname, num_gram, feature_count_threshold, token_dict=None, HDFS_RETR_DIR=None, remove_duplicated="N"
    , cust_featuring=None, cust_featuring_params=None, local_out_dir=None, filter_ratio=None
    ): 

    # zip func in other files for Spark workers ================= ================
    zip_file_path=ml_util.ml_build_zip_file(zipout_dir, zipcode_dir, zip_file_name, user_custom=cust_featuring)
    # get_spark_context
    sc=ml_util.ml_get_spark_context(sp_master
        , spark_rdd_compress
        , spark_driver_maxResultSize
        , sp_exe_memory
        , sp_core_max
        , jobname
        , [zip_file_path]) 
    # log time ================================================================ ================
    t0 = time()

    # input filename
    input_filename="*"
    ext_type='.gz'
    gz_list=None
    convert2dirty="N"
    if not ',' in hdfs_dir_list: # single dir having *.gz ==== =========
        # read raw data from HDFS as .gz format ========== 
        rdd_files=os.path.join(hdfs_dir_list, input_filename+ext_type)
        # check if gz files in hdfs ============
        try:
            gz_list=hdfs.ls(hdfs_dir_list)
            print "INFO: check hdfs folder=",hdfs_dir_list

        except IOError as e:
            print "WARNING: I/O error({0}): {1}".format(e.errno, e.strerror)
        except:
            print "WARNING: Error at checking HDFS file:", sys.exc_info()[0]     
        # use whole folder
        if gz_list is None or len(gz_list)==0:
            rdd_files=hdfs_dir_list
            print "ERROR: No file found by ",input_filename+ext_type #,", use",hdfs_dir_list,"instead"    
            return -2
    else: # multiple dirs ==== =========
        rdd_files=""
        cnt=0
        temp_lbl_list=[]
        comma=""
        print "INFO: before label_arr=",label_arr
        
        # check each folder
        for dr in hdfs_dir_list.split(','):
            #print "****=",dr
            if not len(dr)>0:
                continue
            try:
                # remove space etc.
                dr=dr.strip()
                fdr=os.path.join(HDFS_RETR_DIR, dr)
                #print "fdr=",fdr
                # ls didn't like "*"
                if '*' in fdr:
                    #gz_list=hdfs.ls(fdr.replace("*",""))
                    dn=os.path.dirname(fdr).strip()
                    bn=os.path.basename(fdr).strip()
                    #print "dn=",dn,",bn=",bn
                    # get all names under folder and do filtering
                    gz_list=fnmatch.filter(hdfs.ls(dn), '*'+bn)
                    #print "gz_list=",gz_list
                else:
                    gz_list=hdfs.ls(fdr)
                cnt=cnt+len(gz_list)
                
                if len(gz_list)>0:
                    rdd_files=rdd_files+comma+fdr
                    comma=","
            except IOError as e:
                print "WARNING: I/O error({0}): {1}".format(e.errno, e.strerror)
            except:
                print "WARNING: Error at checking HDFS file:", sys.exc_info()[0]     
        # use whole folder
        if cnt is None or cnt==0:
            print "ERROR: No file found at",rdd_files
            return -2
        else:
            print "INFO: total file count=",cnt
        # set convert flag only when multiple dir and label_arr has dirty label
        #if label_arr is None: # create label arr if None
        #    label_arr=temp_lbl_list
        if not label_arr is None and len(label_arr)==2 and label_arr[1]=="dirty":
            convert2dirty="Y"
    print "INFO: rdd_files=",rdd_files

    txt_rdd=sc.textFile(rdd_files)#, use_unicode=False
    
    total_input_count=txt_rdd.count()
    print "INFO: Total input sample count=",total_input_count
    # debug only
    #for x in txt_rdd.collect():
    #    print "t=",x
    print "INFO: hdfs_dir_list=",hdfs_dir_list
    print "INFO: label_arr=",label_arr
    print "INFO: feature_count_threshold=",feature_count_threshold
    
    #jkey_dict={"meta_list":["label","md5","mdate"], "data_key":"logs"}
    #   this dict depends on the format of input data
    if not data_field_list is None:
        jkey_dict=json.loads(jkey_dict)
        
        data_key=jkey_dict["data_key"]
        meta_list=jkey_dict["meta_list"]
        
        metadata_count=len(meta_list)
        data_idx=metadata_count
        print "INFO: jkey_dict=",jkey_dict
        print "INFO: meta_list=",meta_list
        print "INFO: data_key=",data_key
        print "INFO: data_field_list=",data_field_list
        print "INFO: metadata_count=",metadata_count

        featured_rdd = txt_rdd \
            .map(lambda x: preprocess_json(x,meta_list,data_key,data_field_list)) \
            .filter(lambda x: len(x) > metadata_count) \
            .filter(lambda x: type(x[metadata_count]) is list) \
            .map(lambda x: feature_extraction_ngram(x, data_idx, MAX_FEATURES, num_gram)) \
            .filter(lambda x: len(x) > metadata_count) \
            .filter(lambda x: type(x[metadata_count]) is dict) \
            .filter(lambda x: type(x[metadata_count+1]) is dict) \
            .filter(lambda x: len(x[metadata_count])> int(feature_count_threshold) ) \
            .cache()
            
        #print "INFO: featured_rdd="
        #for x in featured_rdd.collect():
        #    print "INFO: **** f=",x
    # user custom code for featuring  ============================================= ==========
    #   input txt_rdd format (string):  each text row for each sample 
    #   output featured_rdd format (list):[meta-data1,meta-data2,..., hash_cnt_dic, hash_str_dic]
    elif not cust_featuring is None and len(cust_featuring)>0:
        user_module=None
        user_func=None
        user_func_dnn=None
        # load user module =======
        try:
            modules = map(__import__, [CUSTOM_PREFIX+cust_featuring])
            user_module=modules[0]
            user_func=getattr(user_module,CUSTOM_FUNC)
        except Exception as e:
            print "ERROR: module=",CUSTOM_PREFIX+cust_featuring
            print "ERROR: user module error.", e.__doc__, e.message
            return -101
        try:
            jparams=json.loads(cust_featuring_params)
            if jparams and 'n-gram' in jparams:
                num_gram=jparams['n-gram']
            elif jparams and 'ngram' in jparams:
                num_gram=jparams['ngram']
        except Exception as e:
            print "ERROR: user params error.", e.__doc__, e.message
            return -200    
            
        # convert feast into array. output format: [ meta1,meta2,..., [feat1,feat2,...]]   
        tmp_rdd = txt_rdd.map(lambda x: user_func(x, cust_featuring_params)) \
            .filter(lambda x: len(x) > metadata_count) \
            .filter(lambda x: type(x[metadata_count]) is list).cache()
        print " tmp_rdd cnt=", tmp_rdd.count(),",ix=",data_idx,",max f=",MAX_FEATURES,"ngram=",num_gram
        print "take(1) rdd=",tmp_rdd.take(1)
		
		# TBD for multivariant output format: [ meta1,meta2,..., [[feat1,feat2,...],[feat1,feat2,...],...]]
		
		# TBD only for num_gram available
        # for traditional ML, feat in a dict 
		# output format: [ meta1,meta2,..., [[feat1,feat2,...],[feat1,feat2,...],...]]
        featured_rdd = tmp_rdd \
            .map(lambda x: feature_extraction_ngram(x, data_idx, MAX_FEATURES, num_gram)) \
            .filter(lambda x: len(x) > metadata_count) \
            .filter(lambda x: type(x[metadata_count]) is dict) \
            .filter(lambda x: type(x[metadata_count+1]) is dict) \
            .filter(lambda x: len(x[metadata_count])> int(feature_count_threshold) ) \
            .cache()
        #print " f_rdd cnt=", featured_rdd.count()      
        #print featured_rdd.take(1)
        all_hashes_cnt_dic=None
        all_hash_str_dic=None
        all_hashes_seq_dic = None
    else:
        print "INFO: pattern_str=",pattern_str+"<--"
        print "INFO: ln_delimitor=",ln_delimitor+"<--"
        print "INFO: label_idx=",label_idx
        print "INFO: data_idx=",data_idx
        print "INFO: metadata_count=",metadata_count
        print "INFO: filter_ratio=",filter_ratio        
        
        # filter top and least percentage of feature
        if not filter_ratio is None and filter_ratio > 0 and filter_ratio <1:
            # check total count here before continue
            upper_cnt=total_input_count*(1-filter_ratio)
            lower_cnt=total_input_count*filter_ratio
            # set limit for lower bound. if total count is large, lower_cnt may exclude all features...
            # max lower count =  min( MAX_FILTER_LOWER_CNT, total_input_count/100 ) 
            if not MAX_FILTER_LOWER_CNT is None and lower_cnt > MAX_FILTER_LOWER_CNT:
                if MAX_FILTER_LOWER_CNT > total_input_count/100:
                    lower_cnt=total_input_count/100
                else:
                    lower_cnt=MAX_FILTER_LOWER_CNT


            print "INFO: filtering by count, upper bound=",upper_cnt,",lower bound=",lower_cnt
            # find unique feature, count them, remove them if in highest and lowest % and then create a dict 
            f_feat_set = Set (txt_rdd.map(lambda x:x.split(ln_delimitor)).flatMap(lambda x:Set(x[metadata_count:])) \
                .map(lambda x:(x,1)).reduceByKey(lambda a, b: a + b) \
                .filter(lambda x:x[1]<= upper_cnt and x[1]>= lower_cnt) \
                .map(lambda x:x[0]).collect() )
                
            print "INFO: f_feat_set len=",len(f_feat_set)
            broadcast_f_set = sc.broadcast(f_feat_set)

            #txt_rdd=txt_rdd.map(lambda x: filter_by_list(x, metadata_count,ln_delimitor, broadcast_f_list.value ))
            txt_rdd=txt_rdd.map(lambda x: x.split(ln_delimitor)) \
                        .map(lambda x: x[:metadata_count]+ [w for w in x[metadata_count:] if w and w in broadcast_f_set.value]) \
                        .map(lambda x: ln_delimitor.join(x))
        
        
        # preprocess by pattern matching and then extract n-gram features   #.encode('UTF8')
        #   input txt_rdd format (string):  meta-data1\tmeta-data2\t...\tdataline1\tdataline2\t...datalineN\n
        #   output featured_rdd format (list):[meta-data1,meta-data2,..., hash_cnt_dic, hash_str_dic]
        #       hash_cnt_dic: {hash,hash:count,...}  hash_str_dic: {hash: 'str1',... }
        tmp_rdd = txt_rdd \
            .map(lambda x: preprocess_pattern(x, metadata_count, pattern_str, ln_delimitor \
                                                , label_idx, label_arr, convert2dirty )) \
            .filter(lambda x: len(x) > metadata_count) \
            .filter(lambda x: type(x[metadata_count]) is list) #.cache() memory issue...
        #tmp_rdd_count=tmp_rdd.count()
        #print "INFO: After preprocessing count=",tmp_rdd_count
        featured_rdd = tmp_rdd \
            .map(lambda x: feature_extraction_ngram(x, data_idx, MAX_FEATURES, num_gram)) \
            .filter(lambda x: len(x) > metadata_count) \
            .filter(lambda x: type(x[metadata_count]) is dict) \
            .filter(lambda x: type(x[metadata_count+1]) is dict) \
            .filter(lambda x: len(x[metadata_count])> int(feature_count_threshold) ) \
            .cache()
        #feat_rdd_count=featured_rdd.count()
        #print "INFO: After featuring count=",feat_rdd_count

        all_hashes_cnt_dic=None
        all_hash_str_dic=None
        all_hashes_seq_dic = None
    
    #get all hashes and total occurring count ===============
    #   all_hashes_cnt_dic: {'hash,hash': total count,... }
    if all_hashes_cnt_dic is None:
        #all_hashes_cnt_dic = featured_rdd.map(lambda x: x[metadata_count]).reduce(lambda a, b: combine_dic_cnt(a, b))
        all_hashes_cnt_dic = dict(featured_rdd.flatMap(lambda x: x[metadata_count].items()).reduceByKey(lambda a, b: a + b).collect())
    
    #get all hashes and their extracted string  ===============
    #   all_hash_str_dic: {hash:'str1', ...
    if all_hash_str_dic is None:
        #all_hash_str_dic = featured_rdd.map(lambda x: x[metadata_count+1]).reduce(lambda a, b: combine_dic(a, b))
        all_hash_str_dic = dict(featured_rdd.flatMap(lambda x: x[metadata_count+1].items()).distinct().collect())
    
    # get all labels into an array  =============== provided by parameter?
    if label_arr is None:
        # will force "clean" be 0 here
        label_arr=sorted(featured_rdd.map(lambda x: x[label_idx].lower()).distinct().collect())
        # debug only
        print "INFO: label_arr.=",json.dumps(sorted(label_arr))
    
    # save labels to hdfs as text file==================================== ============
    hdfs_folder = hdfs_feat_dir #+ "/"   # "/" is needed to create the folder correctly
    print "INFO: hdfs_folder=", hdfs_folder
    try:
        hdfs.mkdir(hdfs_folder)
    except IOError as e:
        print "WARNING: I/O error({0}): {1}".format(e.errno, e.strerror)
    except:
        print "WARNING: Unexpected error at mkdir:", sys.exc_info()[0]     
    
    # clean up metadata_file
    metadata_file = os.path.join(hdfs_folder , metadata) #"metadata"
    print "INFO: metadata_file=", metadata_file
    try:
        hdfs.rmr(metadata_file)
    except IOError as e:
        print "WARNING: I/O error({0}): {1}".format(e.errno, e.strerror)
    except:
        print "WARNING: Unexpected error at rmr():", sys.exc_info()[0]     
    sc.parallelize(label_arr,1).saveAsTextFile(metadata_file)
    
    #remap all hash values to continuous key/feature number ==============
    #     all_hashes_seq_dic: { hash : sequential_numb }
    if all_hashes_seq_dic is None:
        all_hashes_seq_dic={}
        remap2seq(all_hashes_cnt_dic, all_hashes_seq_dic)   #all_hashes_seq_dic has continuous key number
    #print "all_hashes_seq_dic=",all_hashes_seq_dic
    total_feature_numb=len(all_hashes_seq_dic)
    print "INFO: Total feature count=", len(all_hashes_seq_dic)

    # featured_rdd (list):    [meta-data1,meta-data2,..., hash_cnt_dic, hash_str_dic]
    # seq_featured_rdd(list): [meta-data1,meta-data2,..., hash_cnthsh_dict, hash_str_dic] (feat id in sorted sequence)
    # hash_cnt_dic: {hash: count}  hash_str_dic: {hash: 'str1,str2...' }
    #     set binary_flag to True, all feature:value will be 1
    broadcast_dic = sc.broadcast(all_hashes_seq_dic)
    seq_featured_rdd = featured_rdd.map(lambda x: convert2seq(x,label_idx,data_idx,broadcast_dic.value,binary_flag= True)).cache() 
    
    # get hash_cnthsh_dict then flatMap and reduce to (feat id, count)
    ct_rdd=seq_featured_rdd.flatMap(lambda x: [(i[0],i[1]) for i in x[data_idx].iteritems()]).reduceByKey(lambda a, b: a + b)
    # sorted by feature id as int
    feat_sample_count_arr=ct_rdd.sortBy(lambda x:int(x[0])).map(lambda x:x[1]).collect()
    # sort after collect may fail when rdd is huge
    #feat_sample_count_arr=[]
    #for i in sorted(ct_rdd.collect(), key=lambda t: int(t[0])):
    #    feat_sample_count_arr.append(i[1])
    print "INFO: feat_sample_count_arr len=",len(feat_sample_count_arr)
    
    # save feat_sample_count_arr data ==================================== ============
    filter='{"rid":'+row_id_str+',"key":"feat_sample_count_arr"}'
    upsert_flag=True
    jo_insert={}
    jo_insert["rid"]=eval(row_id_str)
    jo_insert["key"]="feat_sample_count_arr"
    jo_insert["value"]=feat_sample_count_arr
    jstr_insert=json.dumps(jo_insert)
    ret=query_mongo.upsert_doc_t(mongo_tuples,filter,jstr_insert,upsert_flag)
    print "INFO: Upsert count for feat_sample_count_arr=",ret
    # insert failed, save to local
    if ret==0:
        # drop old record in mongo
        ret=query_mongo.delete_many(mongo_tuples,None,filter)
        if not os.path.exists(local_out_dir):
            os.makedirs(local_out_dir)
        fsca_hs=os.path.join(local_out_dir,row_id_str,row_id_str+"_feat_sample_count_arr.pkl")
        print "WARNING: save feat_sample_count_arr to local"
        ml_util.ml_pickle_save(feat_sample_count_arr, fsca_hs)
   
    # save feature data; TBD. not used. ==================================== ============
    
    #libsvm_rdd=seq_featured_rdd.map(lambda x: convert2libsvm(x,label_idx,data_idx,label_arr))
    # put hash to the front of each row, assume hash is after label
    libsvm_rdd=seq_featured_rdd.map(lambda x: x[label_idx+1]+" "+convert2libsvm(x,label_idx,data_idx,label_arr))
    # debug only
    #print "libsvm_rdd="
    #for i in libsvm_rdd.collect():
    #    print i

    # get rdd statistics info
    stats= featured_rdd.map(lambda p: len(p[metadata_count])).stats()
    feat_count_max=stats.max()
    feat_count_stdev=stats.stdev()
    feat_count_mean=stats.mean()
    sample_count=stats.count()
    print "INFO: libsvm data: sample count=",sample_count,",Feat count mean=",feat_count_mean,",Stdev=",feat_count_stdev
    print "INFO:   ,max feature count=",feat_count_max
    # find sample count
    lbl_arr=featured_rdd.map(lambda x: (x[label_idx],1)).reduceByKey(add).collect()
    print "INFO: Sample count by label=",lbl_arr

    
    # remove duplicated libsvm string; only keep the first duplicated item, assume space following key_idx
    if remove_duplicated=="Y":
        libsvm_rdd=libsvm_rdd \
            .map(lambda x: ( ','.join(x.split(' ')[metadata_count:]), x)) \
            .groupByKey().map(lambda x: list(x[1])[0] ) \
            .cache()        
        cnt_list= libsvm_rdd.map(lambda x: (x.split(' ')[1],1)).reduceByKey(add).collect()
        stats= libsvm_rdd.map(lambda x: len(x.split(' ')[metadata_count:])).stats()
        feat_count_max=stats.max()
        feat_count_stdev=stats.stdev()
        feat_count_mean=stats.mean()
        sample_count=stats.count()
        print "INFO: Non-Duplicated libsvm data: sample count=",sample_count,",Feat count mean=",feat_count_mean,",Stdev=",feat_count_stdev
        print "INFO:   ,max feature count=",feat_count_max
        print "INFO: Non-Duplicated Label count list=",cnt_list
        
    # clean up libsvm data ==================================== ============
    libsvm_data_file = os.path.join(hdfs_folder , libsvm_alldata_filename) #"libsvm_data"
    print "INFO: libsvm_data_file=", libsvm_data_file
    try:
        #hdfs.ls(save_dir)
        #print "find hdfs folder"
        hdfs.rmr(libsvm_data_file)
        #if num_gram == 1: 
        #   hdfs.rmr(dnn_data_file)
        #print "all files removed"
    except IOError as e:
        print "WARNING: I/O error({0}): {1} at libsvm_data_file clean up".format(e.errno, e.strerror)
    except:
        print "WARNING: Unexpected error at libsvm file clean up:", sys.exc_info()[0]     
    
    #codec = "org.apache.hadoop.io.compress.GzipCodec"
    #libsvm_rdd.saveAsTextFile(libsvm_data_file, codec)  
    libsvm_rdd.saveAsTextFile(libsvm_data_file) # TBD encrypted
    
    feat_count_file = libsvm_data_file+"_feat_count"
    print "INFO: feat_count_file=", feat_count_file
    try:
        hdfs.rmr(feat_count_file)
    except IOError as e:
        print "WARNING: I/O error({0}): {1} at feat_count clean up".format(e.errno, e.strerror)
    except:
        print "WARNING: Unexpected error at libsvm feature count clean up:", sys.exc_info()[0]     
    sc.parallelize([total_feature_numb],1).saveAsTextFile(feat_count_file)


    label_dic = {}
    # assign label a number
    for idx, label in enumerate(sorted(label_arr)):
        if not label in label_dic:
            label_dic[label] = idx      #starting from 0, value = idx, e.g., clean:0, dirty:1
    
    # output text for DNN:[meta-data1,meta-data2,..., [feature tokens]] ================= DNN ===========
    if num_gram == 1: # special flag to tokenize and keep input orders
        print "INFO: processing data for DNN..."
        # create token dict
        # str_hash_dict: string to hash
        # all_hashes_seq_dic: hash to seq id
        if token_dict is None or len(token_dict)==0:
            token_dict={}
            str_hash_dict={v: k for k, v in all_hash_str_dic.iteritems()}
            for k,v in str_hash_dict.iteritems():
                token_dict[k]=int(all_hashes_seq_dic[str(v)])
            #print "token_dict=",len(token_dict),token_dict
        
        dnn_rdd = tmp_rdd \
            .map(lambda x: tokenize_by_dict(x, data_idx, token_dict,label_idx, label_dic)) \
            .filter(lambda x: len(x) > metadata_count) \
            .filter(lambda x: type(x[metadata_count]) is list) 
            #.cache()
            # filter duplication here
        #print dnn_rdd.take(3)
        
        dnn_data_file = os.path.join(hdfs_folder , dnn_alldata_filename) #"dnn_data"
        print "INFO: dnn_data_file=", dnn_data_file
        try:
            hdfs.rmr(dnn_data_file)
        except IOError as e:
            print "WARNING: I/O error({0}): {1} at dnn_data_file clean up".format(e.errno, e.strerror)
        except:
            print "WARNING: Unexpected error at libsvm file clean up:", sys.exc_info()[0]
        
        # clean up data
        dnn_npy_gz_file=os.path.join(hdfs_folder , row_id_str+"_dnn_")
        print "INFO: dnn_npy_gz_file=", dnn_npy_gz_file
        try:
            hdfs.rmr(dnn_npy_gz_file+"data.npy.gz")
            hdfs.rmr(dnn_npy_gz_file+"label.npy.gz")
            hdfs.rmr(dnn_npy_gz_file+"info.npy.gz")
        except IOError as e:
            print "WARNING: I/O error({0}): {1} at dnn_npy clean up".format(e.errno, e.strerror)
        except:
            print "WARNING: Unexpected error at dnn_npy file clean up:", sys.exc_info()[0]
        # save new data
        try:
            dnn_rdd.saveAsTextFile(dnn_data_file)
        except:
            print "WARNING: Unexpected error at saving dnn data:", sys.exc_info()[0]
        # show data statistics        
        try:
            stats= dnn_rdd.map(lambda p: len(p[metadata_count])).stats()
            feat_count_max=stats.max()
            feat_count_stdev=stats.stdev()
            feat_count_mean=stats.mean()
            sample_count=stats.count()
            print "INFO: DNN data: sample count=",sample_count,",Feat count mean=",feat_count_mean,",Stdev=",feat_count_stdev
            print "INFO:   ,max feature count=",feat_count_max
        except:
            print "WARNING: Unexpected error at getting stats of dnn_rdd:", sys.exc_info()[0]

        
    
    # clean up pca data in hdfs ============ ========================
    pca_files= '*'+libsvm_alldata_filename+"_pca_*"
    #print "INFO: pca_files=", pca_files
    try:
        f_list=hdfs.ls(hdfs_folder)
        if len(f_list)>0:
            df_list=fnmatch.filter(f_list,pca_files)
            for f in df_list:
                print "INFO: rm ",f
                hdfs.rmr(f)
    except IOError as e:
        print "WARNING: I/O error({0}): {1}".format(e.errno, e.strerror)
    except:
        print "WARNING: Unexpected error at libsvm pca file clean up:", sys.exc_info()[0]     

    # clean up pca data in web local ============ ========================       
    pca_fname=os.path.join(model_data_folder , row_id_str+'_pca_*.pkl*')
    print "INFO: pca_fname=", pca_fname
    
    try:
        for fl in glob.glob(pca_fname):
            print "INFO: remove ", fl
            os.remove(fl)
    except OSError, e:
        print ("Error: %s - %s." % (e.pca_fname,e.strerror))
    
    # insert into mongoDB; all_hash_str_dic:  {hash:'str1', ... }
    filter='{"rid":'+row_id_str+',"key":"dic_hash_str"}'
    upsert_flag=True
    jo_insert={}
    jo_insert["rid"]=eval(row_id_str)
    jo_insert["key"]="dic_hash_str"
    jo_insert["value"]=all_hash_str_dic
    jstr_insert=json.dumps(jo_insert)
    #print "jstr_insert=",jstr_insert
    ret=query_mongo.upsert_doc_t(mongo_tuples,filter,jstr_insert,upsert_flag)
    print "INFO: Upsert count for hash_str_dic=",ret
    # insert failed, save to local
    if ret==0:
        # drop old record in mongo
        ret=query_mongo.delete_many(mongo_tuples,None,filter)
        if not os.path.exists(local_out_dir):
            os.makedirs(local_out_dir)
        fn_hs=os.path.join(local_out_dir,row_id_str,row_id_str+"_dic_hash_str.pkl")
        print "WARNING: save hash_str_dic to local"
        ml_util.ml_pickle_save(all_hash_str_dic, fn_hs)
    
    
    # reverse key value from all_hashes_seq_dic-> all_seq_hashes_dic: { sequential_numb : hash }
    all_seq_hashes_dic = {y:x for x,y in all_hashes_seq_dic.iteritems()}
    
    # insert all_seq_hashes_dic into mongoDB  ================ TBD may over 16Mb limit
    filter='{"rid":'+row_id_str+',"key":"dic_seq_hashes"}'
    upsert_flag=True
    jo_insert={}
    jo_insert["rid"]=eval(row_id_str)
    jo_insert["key"]="dic_seq_hashes"
    jo_insert["value"]=all_seq_hashes_dic
    jstr_insert=json.dumps(jo_insert)
    #print "jstr_insert=",jstr_insert
    ret=query_mongo.upsert_doc_t(mongo_tuples,filter,jstr_insert,upsert_flag)
    print "INFO: Upsert count for seq_hashes_dic=",ret
    # insert failed, save to local
    if ret==0:
        # drop old record in mongo
        ret=query_mongo.delete_many(mongo_tuples,None,filter)
        if not os.path.exists(local_out_dir):
            os.makedirs(local_out_dir)
        fn_sh=os.path.join(local_out_dir,row_id_str,row_id_str+"_dic_seq_hashes.pkl")
        print "WARNING: save all_seq_hashes_dic to local"
        ml_util.ml_pickle_save(all_seq_hashes_dic, fn_sh)

    '''
    label_dic = {}
    # assign label a number
    for idx, label in enumerate(sorted(label_arr)):
        if not label in label_dic:
            label_dic[label] = idx      #starting from 0, value = idx, e.g., clean:0, dirty:1
    '''
    # insert label_dic mongoDB ================ =======
    print "INFO: label_dic=", label_dic
    filter='{"rid":'+row_id_str+',"key":"dic_name_label"}'
    upsert_flag=True
    jstr_insert = '{ "rid":'+row_id_str+',"key":"dic_name_label", "value":['
    for kp in [(k,v) for k,v in label_dic.iteritems()] :
        jstr_insert+='{"'+kp[0]+'":'+str(kp[1])+'},'
    # remove last ','
    jstr_insert= jstr_insert[:len(jstr_insert)-1]   
    jstr_insert+=']}'
    #print "INFO: jstr_insert=",jstr_insert
    ## write to mongoDB.myml.dataset_info, ignore doc with duplicated key
    # db.dataset_info.createIndex({"rid":1,"key":1},{unique:true})
    ret=query_mongo.upsert_doc_t(mongo_tuples,filter,jstr_insert,upsert_flag)
    print "INFO: Upsert count for name mapping=",ret

    # clean up excluded feature list from mongoDB ================ =======
    jstr_filter='{"rid":'+row_id_str+',"key":"feature_excluded"}'
    ret=query_mongo.delete_many(mongo_tuples,None, jstr_filter)
    print "INFO: Delete feature_excluded=",ret
    
    
    # only update db for web request ================ =======
    class_numb=len(label_dic)
    if fromweb=="1": 
        dataset_info={"dataset_count":sample_count, "class_count":class_numb ,"training_fraction":"N/A"}
        # clean up PCA opts, needs to be regenerated
        str_sql="UPDATE atdml_document set class_numb = '"+str(class_numb) \
            +"', total_feature_numb='"+str(total_feature_numb) \
            +"', dataset_info='"+json.dumps(dataset_info) \
            +"', ml_pca_opts=null" \
            +", label_arr='"+ json.dumps(sorted(label_arr))+"'" \
            +", ml_n_gram='"+ str(num_gram)+"'" \
            +" where id="+row_id_str
        #print "str_sql=",str_sql
        ret=exec_sqlite.exec_sql(str_sql)
        print "INFO: Data update done! ret=", str(ret),", class_numb=",str(class_numb),json.dumps(sorted(label_arr))
    else:
        print "INFO: class_numb = "+str(class_numb)
    
    
    t1 = time()
    print 'INFO: running time: %f' %(t1-t0)
    return 0
 
# filter by dict to reduce features
def filter_by_list(x, metadata_count, delimitor, f_feat_list ):
    #ret=""
    arr=x.split(delimitor)
    ret=arr[:metadata_count]
    for s in arr[metadata_count:]:
        if s in f_feat_list:
            #ret=ret+sep+s
            ret.append(s)
    #return ret
    return '\t'.join(ret)
 
# convert RDD element to libsvm format
# label_arr: ["clean","dirty"]
# output: 0 1:1 3:1
def convert2libsvm(str_arr,label_idx,data_idx,label_arr=["clean","dirty"]):
    hash_cnt_dict=str_arr[data_idx]
    label=str_arr[label_idx].lower() # all label are lower case?
    ret=""
    # find label
    max_label=str(len(label_arr))
    for idx, val in enumerate(label_arr):
        if val.lower()==label:
            ret=str(idx)+" "
            break
    if ret=="":
        ret=max_label+" " # default to len

    # need to sort the key/feature id for libsvm
    od = collections.OrderedDict(sorted(hash_cnt_dict.items(), key=lambda t: int(t[0])))
    # convert to libsvm
    ret = ret+ json.dumps(od).replace(" ","").replace("{","").replace("}","").replace("\"","").replace(","," ")
    
    return ret
    
# convert RDD element to sequential key/feature id output {seq:cnt}
#   str_arr: [meta-data1,meta-data2,...,meta-dataN, hash_cnthsh_dic, hash_str_dic]
def convert2seq(str_arr, label_idx, data_idx, all_hashes_seq_dic, binary_flag):
    hash_cnthsh_dict=str_arr[data_idx]
    out={}
    
    # set new feature id
    for key in hash_cnthsh_dict:
        if binary_flag:
            out[all_hashes_seq_dic[key]]=1  #force to 1; #
        else:
            out[all_hashes_seq_dic[key]]=hash_cnthsh_dict[key]
    str_arr[data_idx]=out
    return str_arr
    
# combine two dictionaries into one dictionary; for Spark Reduce()   
def combine_dic(dic_1, dic_2):
    return dict(dic_1.items()+dic_2.items())

    # combine two dictionaries into one dictionary and add count; for Spark Reduce()   
def combine_dic_cnt(dic_1, dic_2):
    combine_dic = dic_1.copy()
    for key in dic_2:
        if key in combine_dic:
            combine_dic[key]=dic_2[key]+combine_dic[key] # 
        else:
            combine_dic[key]=dic_2[key]
    return combine_dic


# map hash/key to continuous number for input_dic ================          
def remap2seq(input_dic, output_dic):
    cnt = 1 # starting from 1, libsvm didn't like 0
    for key in input_dic:
        output_dic[key] = str(cnt)
        cnt += 1 


if __name__ == '__main__':
    __description__ = "text file preprocessing and feature extraction"
    main()
