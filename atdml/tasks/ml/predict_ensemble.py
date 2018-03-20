#! /usr/bin/python
'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''

# python libraries
import os, stat
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

# our own library imports
sys.path.append('./ml')
sys.path.append('./db')
import query_mongo
#import sqlite3
import exec_sqlite
import ml_util, predict_single_file_pattern, zip_preprocess_pattern
from joblib import Parallel, delayed
import multiprocessing,math

CONF_FILE='../../app.config' # at the base dir of the web
config=ConfigParser.ConfigParser()
config.read(CONF_FILE)
# set CPU count to half of instance
PARALLEL_CNT=int(math.ceil( multiprocessing.cpu_count() ))
print "PARALLEL_CNT=",PARALLEL_CNT

def main():

    parser = ArgumentParser(description=__description__)
    parser.add_argument("-r", "--row_id", type=str, metavar="id for ensemble classifier", help="id for ensemble classifier", required=False)
    parser.add_argument("-i", "--pid_str", type=str, dest='pid_str', metavar="id for prediction record", help="id for prediction record", required=False)
    
    # if id of ensemble classifier not provided, dlist is needed
    parser.add_argument("-dlist", "--ds_list", type=str, metavar="dataset id list of classifier "
        , help="a list of dataset id to retrieve classifier", required=False)
        
    parser.add_argument("-d", "--name", type=str, metavar="file name", help="file name for prediction", required=False)
    parser.add_argument("-o", "--local_out_dir", type=str, metavar="output folder for prediction", help="output folder for prediction", required=False)
    
    ####other parameters
    parser.add_argument("-fw", "--fromweb", type=str, metavar="flag for web", help="flag for web", required=False)

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
    parser.add_argument("-flm", "--flag_local_model", type=str, dest='flag_local_model', help="flag to use local model"
            , default ="Y")

    # not used, for compatibility only
    parser.add_argument("-nb", "--num", type=str, metavar="n gram", help="window size for n gram", required=False)
    parser.add_argument("-pm", "--parameter", type=str, metavar="parameters in json", help="json string contains learning alg and parameter selection", required=False)
    parser.add_argument("-pp", "--pca_param", type=str, metavar="pca parameters in json", help="json string contains pca parameter selection", required=False)
    parser.add_argument("-lb", "--lib", type=str, metavar="spark mllib or scikit", help="learning library used", required=False)
    parser.add_argument("-sl", "--showlabelname", type=str, metavar="show label name", help="0: not shown; 1: show label name", required=False)
    parser.add_argument("-dsid", "--ds_id", type=str, metavar="source dataset id", help="source dataset id for training option", required=False)
                    
    # SPARK
    parser.add_argument('-sp','--sp_master', type=str, dest='sp_master', help='spark.master'
                , default =config.get('spark', 'spark_master'))
    parser.add_argument('-em','--exe_memory', type=str, dest='exe_memory', help='spark.executor.memory'
                , default =config.get('spark', 'spark_executor_memory'))
    parser.add_argument('-cm','--core_max', type=str, dest='core_max', help='spark.cores.max'
                , default =config.get('spark', 'spark_cores_max'))
    
    # database for output
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
    parser.add_argument('-ty','--type', type=str, dest='type', help='action type'
                , default ="predict")
    parser.add_argument('-tid','--tgt_id', type=str, dest='tgt_id', help='target id for ensemble copy script'
                , default =None)
    
    args = parser.parse_args()
    
    if args.name:
        input_gz = args.name
    else:
        input_gz  = None
    # local_out_dir should be pointed to result folder
    if args.local_out_dir:
        local_out_dir = args.local_out_dir
    else:
        local_out_dir  = '.'
    # id for ensemble classifier
    if args.row_id:
        row_id_str = str(args.row_id)
    else:
        row_id_str  = '888'
    if args.pid_str:
        pid_str = str(args.pid_str)
    else:
        pid_str  = '888'
    # list of dataset id for classifier   
    if args.ds_list:
        ds_list=eval(args.ds_list)
        #ds_list=args.ds_list.split(',')
        if isinstance(ds_list, int):
            ds_list=[ds_list]
    else:
        ds_list=None
        
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
        pattern_str  = '(.*)' # 
    if args.ln_delimitor:
        ln_delimitor = args.ln_delimitor
    else:
        ln_delimitor  = '\t'        
    # flag to control outpout
    if args.fromweb:
        fromweb = args.fromweb
    else:
        fromweb  = "1"
    if args.verbose: 
        verbose = args.verbose
    else:
        verbose = "0"
    # database
    if len(args.username)>0:
        username = args.username
    else:
        username  = None
    if len(args.password)>0:
        password = args.password
    else:
        password  = None     

    if args.type == "copy":
        return gen_copy_ensemble_scripts(row_id_str,args.tgt_id)
    

    return predict(row_id_str , pid_str, input_gz, local_out_dir, ds_list
        , fromweb, verbose, label_idx, data_idx, metadata_count, pattern_str, ln_delimitor
        , args.sp_master, args.exe_memory, args.core_max
        , ip_address=args.ip_address, port=args.port, db_name=args.db_name, tb_name=args.tb_name
        , username=username, password=password
        , feat_cnt_threshold=args.feat_cnt_threshold
        , flag_local_model=args.flag_local_model
        )

    
#  Used by massive prediction too ================================= =======================
def predict(row_id_str , pid_str, input_gz, local_out_dir, ds_list
        , fromweb, verbose='0' , label_idx=0, data_idx=3, metadata_count=3, pattern_str=None, ln_delimitor='\t'       
        , sp_master=config.get('spark', 'spark_master'), exe_memory=config.get('spark', 'spark_executor_memory')
        , core_max=config.get('spark', 'spark_cores_max')
        , ip_address=config.get('mongo', 'out_ip_address'), port=eval(config.get('mongo', 'out_port'))
        , db_name=config.get('mongo', 'out_db'), tb_name=config.get('mongo', 'out_tb')
        , username=config.get('mongo', 'out_username'), password=config.get('mongo', 'out_password')
        , feat_cnt_threshold=config.get('machine_learning', 'feature_count_threshold')
        , flag_local_model="Y"
        ):
    t0 = time()
    if ds_list is None or len(ds_list)==0:
        # get ds_list from ensemble record id=row_id_str
        str_sql='select id, ds_list from atdml_document where id='+row_id_str+' and file_type="ensemble" '
        ret=exec_sqlite.query_db(str_sql)
        if not ret is None and len(ret)>0:
            ds_list=eval(ret[0][1])
        if ds_list is None or len(ds_list)==0:
            print "ERROR: ensemble list not found for id=",row_id_str
            return -1  
    print "INFO: classifier count=",len(ds_list) #, ds_list
    
    # expect local_out_dir to result folder; remove rid in path if any
    bname=os.path.basename(local_out_dir)
    if bname.isdigit():
        local_out_dir=os.path.dirname(local_out_dir)
    
    # one spark session for the for loop
    sc=None
    # for each id in ds_list  ======================
    ret_list=[]
    icnt=1
    # local parallel
    if flag_local_model=="Y":
        # get one_line from gz file
        one_line=None
        try:  
            one_line = zip_preprocess_pattern.convert_to_line_by_bash(input_gz,metadata_count,ln_delimitor) # check if one line, if raw file then convert to 1 line
            #print "one_line=",one_line[:100].replace('\t',',')
            #print "one_line=",one_line.replace('\t',',')
            print "INFO: one_line len=", len(one_line)
        except Exception as e:
            print "ERROR: load data file ["+input_gz+"] failed.",e
            return -5
            
        # parallel processing
        ret_list=Parallel(n_jobs=PARALLEL_CNT)(delayed(predict_one) \
            (id,one_line, local_out_dir,metadata_count,ln_delimitor,label_idx, data_idx,feat_cnt_threshold) for id in ds_list)
    
    else: # by Spark
        for id in ds_list:
            #t3 = time()
            pred_ret=None
            # get info from ds id
            (rid, num_gram,ml_opts_str,ds_id_str,lib_mode,option_state,pattern_str,label_arr,ml_opts \
                ,learning_algorithm, file_name) = query_db(id)
                
            if id is None: # no result found for classifier
                print "WARNING: dataset id=",id, "not found!"
                continue

            # get one spark context if needed
            if lib_mode=='mllib' and sc is None:
                sc=predict_single_file_pattern.get_sc(str(id),sp_master, exe_memory, core_max)

            #local_out_dir was used to get model/mapping files
            id_out_dir=os.path.join(local_out_dir,str(id))

            # call prediction(): may need Spark context for mllib
            try:
                #print "before pred",str(id),ds_id_str
                pred_ret= predict_single_file_pattern.predict(
                     row_id_str=str(id), ds_id=ds_id_str
                    ,num_gram=num_gram, j_str=ml_opts_str, lib_mode=lib_mode
                    ,cid_str=str(pid_str) ,input_gz=input_gz, local_out_dir=id_out_dir
                    ,fromweb="2" # force to return json output
                    ,verbose=str(verbose), label_idx=label_idx, data_idx=data_idx, metadata_count=metadata_count
                    ,pattern_str=pattern_str, ln_delimitor=ln_delimitor, feat_cnt_threshold=feat_cnt_threshold
                    , sc=sc)  
                #print "**** ret=",pred_ret
                ret_list.append(pred_ret)
            except:
                print "ERROR: pid=",row_id_str,",classifier id=",id,",msg=", sys.exc_info()[0]
                ret_list.append({"id":int(pid_str),"opt_id":int(id),"ds_id": int(ds_id_str) \
                    ,"prediction":None,"predict_val":None,"learning_algorithm":None \
                    ,"lib":None, "ml_opts":None,"predict_index": None })
            #print "INFO: id=",id,",ret=", ret

            icnt=icnt+1
   
    print "INFO: ret_list len=", len(ret_list)
    
    # TBD find max predict value to pick a label. need review  ====================== ==========
    status="ensemble predicted"
    max_val=sys.maxint * -1
    max_id=0
    alg=None
    lib=None
    prediction=None
    mix_algs=0
    for rt in ret_list:
        alg=rt["learning_algorithm"]
        if not rt is None and "predict_val" in rt and rt["predict_val"] > max_val:
            max_val=rt["predict_val"]
            max_id=rt["opt_id"]
            prediction=rt["prediction"]
            lib=rt["lib"]
        if "svm" in alg:
            mix_algs= mix_algs | 1
        if "logistic" in alg:
            mix_algs= mix_algs | 2
    # check if mixing model types       
    if mix_algs == 3:
        print "WARNING: Mixing SVM and Logistic Regression algorithms found in Ensemble datasets. The prediction may be INVALID!!!"
    outj={"prediction":prediction,"predict_val":max_val,"status":status,"predict_ds":max_id \
        ,"lib":lib,"learning_algorithm":alg,"processed_date":str(datetime.datetime.now()),"returns":ret_list}
    
    result_fname=os.path.join(local_out_dir,pid_str,pid_str+"_predict_output.json")
    #print "INFO: result_fname=",result_fname
    print "RESULT: predict_val=",max_val,",prediction=",prediction
    # create/clean up folder
    ml_util.ml_prepare_output_dirs(pid_str, os.path.join(local_out_dir,pid_str), local_out_dir, result_fname)
    
    # write output to a file
    with open(result_fname,"w") as fo:
        json.dump(outj, fo)
        
    # Update prediction records here     ====================== ==========
    if fromweb=="1": 
        #print "database update"
        str_sql="UPDATE atdml_document set status = '"+status+"', processed_date ='" \
            +str(datetime.datetime.now())+"', prediction = '"+ str(prediction)  \
            +"', predict_val = '"+str(max_val) \
            +"', dataset_info = '"+str(max_id) \
            +"' where id="+pid_str
        ret=exec_sqlite.exec_sql(str_sql)
        
    t1 = time()
    print 'INFO: running time: %f' %(t1-t0)
    return 0

# predict by local model  =====================================================================
def query_db(id):
    # get info from ds id
    str_sql='select id, ml_n_gram, ml_opts, train_id, ml_lib, option_state, ml_pca_opts, pattern' \
            + ', label_arr, filename, file_type,perf_measures,dataset_info ,accuracy,total_feature_numb,ml_pca_opts ' \
            +' from atdml_document where id=' \
            +str(id)
    num_gram=None
    ml_opts_str=None
    lib_mode=None
    ds_id_str=None
    pred_ret=None
    option_state=None
    pattern_str=None
    label_arr=None
    ds_id=None
    ml_opts=None
    learning_algorithm=None
    file_name=None
    file_type=None
    perf_measures=None
    dataset_info=None
    accuracy=None
    total_feature_numb=None
    ml_pca_opts=None
    # get data from sqlite
    ret=exec_sqlite.query_db(str_sql)
    if not ret is None and len(ret)>0:
        num_gram=eval(ret[0][1])
        ml_opts_str=ret[0][2]
        ds_id_str=str(ret[0][3])
        lib_mode=ret[0][4]
        option_state=ret[0][5]
        pattern_str=ret[0][7]
        label_arr=ret[0][8]
        file_name=ret[0][9]
        file_type=ret[0][10]
        perf_measures=ret[0][11]
        dataset_info=ret[0][12]
        accuracy=ret[0][13]
        total_feature_numb=ret[0][14]
        ml_pca_opts=ret[0][15]
        if ds_id_str is None or ds_id_str =="":
            ds_id_str=str(id)
        try:
            ml_opts=json.loads(ml_opts_str)
            learning_algorithm = ml_opts['learning_algorithm'] 
        except:
            print "ERROR: can't get learning_algorithm. ml_opts=",ml_opts_str
            id=None


    # for option_state= new_featuring 
    if option_state=='new_featuring' or ds_id_str is None or len(ds_id_str)==0 or ds_id_str=="None":
        ds_id_str=str(id)

    return (id,num_gram,ml_opts_str,ds_id_str,lib_mode,option_state,pattern_str \
        ,label_arr,ml_opts,learning_algorithm,file_name,file_type,perf_measures,dataset_info,accuracy \
        ,total_feature_numb,ml_pca_opts)
    
# predict by local model file =====================================================================
def predict_one(id,one_line,local_out_dir,metadata_count,ln_delimitor,label_idx, data_idx,feat_cnt_threshold):
 
    # get info from ds id
    (rid, num_gram,ml_opts_str,ds_id_str,lib_mode,option_state,pattern_str,label_arr,ml_opts,
        learning_algorithm,file_name,file_type,perf_measures,dataset_info,accuracy,total_feature_numb,ml_pca_opts) = query_db(id)
    if rid is None: # no result found for classifier
        print "WARNING: dataset id=",id, "not found!"
        return {"id":id,"prediction":""}    
        
    print "INFO: *** Classifier id=",id,",alg=",learning_algorithm,",ds_id_str=",ds_id_str,",lib_mode=" \
        ,lib_mode,",label_arr=",label_arr
        
    #local_out_dir was used to get model/mapping files
    id_out_dir=os.path.join(local_out_dir,str(id))
        
    # find if model exists locally
    mfile=os.path.join(local_out_dir,str(id),str(id)+'_model.json')
    mpfile=os.path.join(local_out_dir,str(id),str(id)+'_model',str(id)+'.pkl')
    jmodel=None
    if os.path.isfile(mfile):
        #print "mfile=",mfile
        #t2 = time()
        #print 'INFO: **** before loading json time 1: %f' %(t2-t3)
        
        with open(mfile) as f:    
            jmodel = json.load(f)
        #print "jmodel len corr=",len(jmodel["coef_arr"])
        coef_arr=jmodel["coef_arr"]
        coef_intercept=jmodel["coef_intercept"]
        dic_hashes_seq=jmodel["dic_hashes_seq"]
        
        dic_lbl={}
        if label_arr is None: #[{"dirty": 1}, {"clean": 0}]
            for i in jmodel["dic_name_label"]:
                dic_lbl=dict(dic_lbl.items() + i.items())

            label_arr=json.dumps(sorted(dic_lbl,key=lambda x: x[1],reverse=True))
        
        feat_arr = predict_single_file_pattern.preprocess_one_line(one_line, num_gram, metadata_count=metadata_count \
                , pattern_str=pattern_str, ln_delimitor=ln_delimitor,label_idx=label_idx, data_idx=data_idx \
                , label_arr=label_arr)
        # get the hash_cnt_dic: {hash,hash:count),...}
        #print "2=",feat_arr[2],metadata_count
        hashes_cnt_dic=feat_arr[metadata_count]
        #print "hashes_cnt_dic len=",len(hashes_cnt_dic)
        hypothesis_val=0
        
        #t2 = time()
        #print 'INFO: **** after preprocess_raw_file 3: %f' %(t2-t3)
        nfcnt=0
        #curr_dic={}
        for hashes in hashes_cnt_dic: # 
            if hashes in dic_hashes_seq :
                # assume binary value (0/1) for feature vector;dic_hashes_seq is 1 based, coef_arr is 0 based
                #curr_dic[dic_hashes_seq[hashes] ]=1
                hypothesis_val=hypothesis_val+round(coef_arr[dic_hashes_seq[hashes]-1],11)
                #print "INFO: f=",(dic_hashes_seq[hashes]-1),hashes#, coef_arr[dic_hashes_seq[hashes]-1]
            else:
                #print "INFO: Feature '"+hashes+"' not found"
                nfcnt=nfcnt+1
        
        hypothesis_val=hypothesis_val+coef_intercept
        
        # check feature count
        feat_cnt=len(hashes_cnt_dic)
        f_not_found=0

        print "INFO: feature found count=",feat_cnt - nfcnt,", feat_cnt_threshold=",feat_cnt_threshold
        if (feat_cnt - nfcnt) < int(feat_cnt_threshold):
            print "WARNING: feature count for this sample="+str(feat_cnt- nfcnt)+" is less than threshold="+str(feat_cnt_threshold)+""
            f_not_found=1

        sing_label_pred=0
        pred_label=""
        if learning_algorithm and "logistic" in learning_algorithm.lower():
            hypothesis_val=ml_util.sigmoid(hypothesis_val) 
            if hypothesis_val>=0.5:
                sing_label_pred=1
        else:
            if hypothesis_val>=0:
                sing_label_pred=1
        
        # check threshold & set label===========
        if f_not_found == 1:
            pred_label="not_enough_info"
            status="new"
        elif not sing_label_pred is None and not label_arr is None:
            label_arr=eval(label_arr)
            pred_label = label_arr[int(sing_label_pred)]
            
        print "INFO: id=",id,file_name,",f cnt=",feat_cnt,",non-exist=",nfcnt,",hwx=",hypothesis_val,",pred_label=",pred_label
        return {"id":int(id),"opt_id":int(id),"ds_id": int(ds_id_str) \
            ,"prediction":str(pred_label),"predict_index": int(sing_label_pred) \
            ,"predict_val":hypothesis_val,"learning_algorithm":learning_algorithm \
            ,"lib":lib_mode, "ml_opts":ml_opts }

# generate scripts to copy ensemble datasets  =====================================================================
#       Sql to insert dataset info; script to create web folders locally
def gen_copy_ensemble_scripts(src_id, tgt_id, fout_sql="_id_migrate.sql", fout_bash="_id_migrate_bash.sh",hn_subfix=".?your.com"
        , dir_prefix=config.get('app', 'TRAIN_DES_DIR')):
    #print "in copy_ensemble()"
    str_sql='select id,ds_list,docfile,filename,status,status_code,submitted_by,file_type' \
        + ', acl_list, desc,created_date  from atdml_document where id=' +str(src_id)
    # get data from sqlite
    ret=exec_sqlite.query_db(str_sql)
    tgt_id=int(tgt_id)
    i=0
    base_sql= "insert into atdml_document(id,docfile,filename,status,status_code,submitted_by,file_type,acl_list,desc,created_date,ds_list)" \
    "values ('%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','_xxx_');\n" %  \
    (tgt_id,ret[0][2],ret[0][3],ret[0][4],ret[0][5],ret[0][6],ret[0][7],ret[0][8],ret[0][9],ret[0][10])
    #ds_id=tgt_id
    acl_list=ret[0][8]
    created_date=ret[0][10]
    desc="model "
    out_sql_f=None 
    out_bash_f=None 
    
    # open file for write
    if not fout_sql is None and len(fout_sql)>0:
        fout_sql=fout_sql.replace("_id_",str(src_id))
        out_sql_f = open(fout_sql, 'w')
    if not fout_bash is None and len(fout_bash)>0:
        fout_bash=fout_bash.replace("_id_",str(src_id))
        out_bash_f = open(fout_bash, 'w')
        os.chmod(fout_bash, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        out_bash_f.write("cp -rp "+dir_prefix+"/"+str(src_id)+" "+dir_prefix+"/copy/"+str(tgt_id)+" && rename '" \
                +str(src_id)+"' '"+str(tgt_id)+"' "+dir_prefix+"/copy/"+str(tgt_id)+"/"+str(src_id)+"*.*\n")
        
    tgt_id += 1
    # for ensemble datasets
    if not ret is None and len(ret)>0:
        ds_list=ret[0][1]
        new_ds_list=[]
        comma=''
        # for each id: insert sql, copy files
        for id in ds_list.split(','):
            (id,num_gram,ml_opts_str,ds_id_str,lib_mode,option_state,pattern_str
                ,label_arr,ml_opts,learning_algorithm,file_name,file_type,perf_measures,dataset_info,accuracy,total_feature_numb,ml_pca_opts) =query_db(id)
            if ml_pca_opts is None or ml_pca_opts =="None" or len(ml_pca_opts)==0:
                ml_pca_opts=""
            # sql for sqlite db 
            sql= "insert into atdml_document(id,ml_n_gram, ml_opts, ml_lib,option_state,pattern" \
            ",label_arr,filename, docfile, status, status_code, submitted_by, file_type, acl_list, desc,created_date,perf_measures,dataset_info,accuracy,total_feature_numb,ml_pca_opts)" \
            "values ('%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s');\n" % \
            (tgt_id,num_gram,ml_opts_str, lib_mode, option_state,pattern_str,label_arr,file_name
            ,'','trained','500','copied',file_type,acl_list,desc+str(id),created_date,perf_measures,dataset_info,accuracy,total_feature_numb,ml_pca_opts) 
            if out_sql_f:
                out_sql_f.write(sql);
            # copy files
            bash="cp -rp "+dir_prefix+"/"+str(id)+" "+dir_prefix+"/copy/"+str(tgt_id)+" && rename '" \
                +str(id)+"' '"+str(tgt_id)+"' "+dir_prefix+"/copy/"+str(tgt_id)+"/"+str(id)+"*.*\n"
            if out_bash_f:
                out_bash_f.write(bash);
            new_ds_list.append(str(tgt_id)) 
            comma=','
            tgt_id += 1
        strlist=','.join(new_ds_list)
        base_sql=base_sql.replace('_xxx_', strlist)
        if out_sql_f:
            out_sql_f.write(base_sql)
            out_sql_f.close()
        else:
            print base_sql
        if out_bash_f:
            out_bash_f.close()
            
if __name__ == '__main__':
    __description__ = "single file prediction by ensemble"
    main()
