#! /usr/bin/python
'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''


# standard library imports
from argparse import ArgumentParser
import sys, ConfigParser, pickle
import os, glob
import re
import collections, gzip, mimetypes
import ujson, json, math, zipfile
import numpy as np
from scipy.sparse import csr_matrix
import subprocess

from scipy.stats import entropy
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pydoop.hdfs as hdfs

from bson import json_util
from sklearn.metrics import roc_curve, auc
#####matplotlib###############
import matplotlib, math
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

#####import for mongodb ####
sys.path.append('./db')
import query_mongo


# may chg to local dir for offline prediction ########################
CONF_FILE='/home/django/myml/app.config' # at the base dir of the web

config=ConfigParser.ConfigParser()
config.read(CONF_FILE)
ROC_DOT_MAX_CNT=7100
BAR_BIN_COUNT=30
CUSTOM_PREFIX='cf_'
CUSTOM_FOLDER='./user_custom'

    
def main():
    parser = ArgumentParser(description=__description__)
    parser.add_argument('-md', '--mode', type = str, metavar = 'utilmode', help = 'action mode', required =False)
    parser.add_argument("-r", "--row_id", type= str, metavar="row id", help="row_id number in the db", required=False)
    parser.add_argument('-ky', '--key', type = str, metavar = 'key', help = 'key for dataset info', required =False)
    parser.add_argument('-dt', '--dict', type = str, metavar = 'dictionary', help = 'dictionary', required =False)
    parser.add_argument('-fn', '--fname', type = str, metavar = 'filename', help = 'filename', required =False)
    parser.add_argument("-dsid", "--ds_id", type=str, metavar="source dataset id", help="source dataset id for training option", required=False)
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
    args = parser.parse_args()
    
    if args.row_id:
        row_id_str = args.row_id
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
        
    if args.mode:
        #print "INFO: in ml_util=" ,args.mode
        if args.mode == "save_dict":
            save_dict(args.row_id, args.key, args.dict
                ,args.ip_address, args.port, args.db_name, args.tb_name, username, password)
        elif args.mode == "save_libsvm_label":
            save_libsvm_label(args.fname , args.row_id, args.key
                ,args.ip_address, args.port, args.db_name, args.tb_name, username, password)
        elif args.mode == "save_libsvm_metadata":
            ml_save_libsvm_metadata(args.fname , args.row_id, args.key
                ,args.ip_address, args.port, args.db_name, args.tb_name, username, password)
        elif args.mode == "build_feat_list":
            build_feat_list(row_id_str, args.fname , None,None,None,args.ds_id
                ,args.ip_address, args.port, args.db_name, args.tb_name, username, password)
        elif args.mode == "get_model":
            ip_address=config.get('mongo', 'out_ip_address')
            #build_feat_list(document 
            #    ,args.ip_address, args.port, args.db_name, args.tb_name, username, password)

            
#============================================================= build_zip_file ==================
# build zip file of all python code for distributing to spark workers, prefix "zip" and sufix ".py"
def ml_build_zip_file(out_dir, code_dir, zip_file_name, user_custom=None, prefix='zip_', suffix='.py'
        , cust_dir=CUSTOM_FOLDER, cust_prefix=CUSTOM_PREFIX):
    zip_file_path = os.path.join(out_dir, zip_file_name)
    with zipfile.ZipFile(zip_file_path, 'w') as pyz:
        rel_dir=""
        for path, dirs, files in os.walk(code_dir):
            rel_dir=path
            for f in files:
                if f.endswith(suffix) and f.startswith(prefix):
                    pyz.write(os.path.join(path, f), os.path.join(os.path.relpath(path, code_dir), f))
        # add user code; set to same ./ml folder in zip file
        if not user_custom is None and len(user_custom)>0:
            for path, dirs, files in os.walk(cust_dir):
                #print path
                for f in files:
                    # include all cf_ files
                    if f.endswith(suffix) and f.startswith(cust_prefix):
                    #if f==cust_prefix+user_custom+suffix:
                        pyz.write(os.path.join(path, f), os.path.join(os.path.relpath(rel_dir, code_dir), f))
            
    #print "INFO: zip_file_path=",zip_file_path    
    return zip_file_path

#============================================================= get_model ==================
# GET model info from mongo for offline prediction
def ml_get_dataset_info(rid, key, ip_address=None, port=None, db_name=None, tb_name=None, username=None, password=None):

    row_id_str=str(rid)
    
    # get db info from config
    if ip_address is None:
        ip_address=config.get('mongo', 'out_ip_address')
    if port is None:
        port=config.get('mongo', 'out_port')
    if db_name is None:
        db_name=config.get('mongo', 'out_db')
    if tb_name is None:
        tb_name=config.get('mongo', 'out_tb')
    
    # get value ===========
    jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
    jstr_proj='{"value":1}'
    try:
        doc=query_mongo.find_one(ip_address, port, db_name, tb_name
                , username, password, jstr_filter, jstr_proj)
        ret = doc['value']   
    except:
        ret=None
    return ret    


    
#============================================================= get_model ==================
# GET model info from mongo for offline prediction
def ml_get_model_t(document_json, mongo_tuples):
    return ml_get_model( document_json
        , mongo_tuples[0], mongo_tuples[1], mongo_tuples[2] 
        , mongo_tuples[3], mongo_tuples[4], mongo_tuples[5])

# GET model info from mongo for offline prediction
# id, filename, file_type, status, local_processed_date, ml_n_gram, ml_lib, ml_opts
#   , accuracy, train_id, option_state
#   , dic_seq_hashes, coef_arr, coef_intercept, pca_param, dic_hash_str, dic_name_label
def ml_get_model(document_json, ip_address=None, port=None, db_name=None, tb_name=None, username=None, password=None):

    rid=document_json["id"]
    train_id=document_json["train_id"]
    option_state=document_json["option_state"]
    row_id_str=str(rid)
    # check if a training option record
    if option_state=="new_training" and not train_id is None:
        row_id_str=str(train_id)
        
    #print "row_id_str=",row_id_str
    
    # get db info from config
    if ip_address is None:
        ip_address=config.get('mongo', 'out_ip_address')
    if port is None:
        port=config.get('mongo', 'out_port')
    if db_name is None:
        db_name=config.get('mongo', 'out_db')
    if tb_name is None:
        tb_name=config.get('mongo', 'out_tb')
    
    # 1. get dic_seq_hashes ===========
    # get {seq :hash,hash } mapping from mongo  key=dic_seq_hashes ===================
    key = "dic_seq_hashes"
    jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
    jstr_proj='{"value":1}'
    dic_hashes_seq={}
    try:
        doc=query_mongo.find_one(ip_address, port, db_name, tb_name
                , username, password, jstr_filter, jstr_proj)
        dic_seq_hashes = doc['value']
        dic_hashes_seq={v: int(k) for k, v in dic_seq_hashes.iteritems()}
    except:
        dic_seq_hashes=None
        print ip_address, port, db_name, tb_name, username, password, row_id_str
        print "WARNING: dic_seq_hashes not found"
    #document_json["dic_seq_hashes"]=dic_seq_hashes
    # allow predict script to find feature id directly.
    document_json["dic_hashes_seq"]=dic_hashes_seq    

    # 2. get coef_arr: always by rid ===========
    key = "coef_arr"
    jstr_filter='{"rid":'+str(rid)+',"key":"'+key+'"}'
    try:
        doc=query_mongo.find_one(ip_address, port, db_name, tb_name
                , username, password, jstr_filter, jstr_proj)
        coef_arr = doc['value']    
    except:
        coef_arr=None
        print "WARNING: coef_arr not found"
    document_json["coef_arr"]=coef_arr    

    # 3. get coef_intercept: always by rid ===========
    key = "coef_intercept"
    jstr_filter='{"rid":'+str(rid)+',"key":"'+key+'"}'
    try:
        print "jstr_filter=",jstr_filter
        doc=query_mongo.find_one(ip_address, port, db_name, tb_name
                , username, password, jstr_filter, jstr_proj)
        coef_intercept = doc['value']    
    except:
        coef_intercept=None
        print "WARNING: coef_intercept not found"
    document_json["coef_intercept"]=coef_intercept    

    # 4. get pca_param:  ===========
    key = "pca_param"
    jstr_filter='{"rid":'+str(rid)+',"key":"'+key+'"}'
    try:
        doc=query_mongo.find_one(ip_address, port, db_name, tb_name
                , username, password, jstr_filter, jstr_proj)
        pca_param = doc['value']    
    except:
        pca_param=None
        print "WARNING: pca_param not found"
    document_json["pca_param"]=pca_param     
    
    # 5. get dic_hash_str ===========
    key = "dic_hash_str"
    jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
    try:
        doc=query_mongo.find_one(ip_address, port, db_name, tb_name
                , username, password, jstr_filter, jstr_proj)
        dic_hash_str = doc['value']
    except:
        dic_hash_str=None
        print "WARNING: dic_hash_str not found"
    document_json["dic_hash_str"]=dic_hash_str    
    
    # 6. get dic_name_label ===========
    key = "dic_name_label"
    jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
    try:
        doc=query_mongo.find_one(ip_address, port, db_name, tb_name
                , username, password, jstr_filter, jstr_proj)
        dic_name_label = doc['value']
    except:
        dic_name_label=None
        print "WARNING: dic_name_label not found"
    document_json["dic_name_label"]=dic_name_label 

    # special for IN =========== =========== =========== ===========
    if " in-dynamic" in document_json["file_type"]:
        # get mapping_dynamic ===========
        key = "mapping_dynamic"
        jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
        try:
            doc=query_mongo.find_one(ip_address, port, db_name, tb_name
                    , username, password, jstr_filter, jstr_proj)
            mapping_dynamic = doc['value']
        except:
            mapping_dynamic=None
            print "WARNING: mapping_dynamic not found"
        document_json["mapping_dynamic"]=mapping_dynamic
    elif " in-static" in document_json["file_type"]:
        # get global variables from MongoDB
        static_list_json={
            "geometry_size_edges":[]
            , "dict_20":{}, "dict_22":{}, "edges_21":[]
            , "edges_23":[], "edges_24":[], "edges_26":[], "dict_45":{}, "dict_47":{}, "all_fps_dic":{}
        } 
        for key in static_list_json:
            jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
            jstr_proj='{"value":1}'
            #print "jstr_filter=",jstr_filter,",jstr_proj=",jstr_proj
            try:
                doc=query_mongo.find_one(ip_address, port, db_name, tb_name
                        , username, password, jstr_filter, jstr_proj)
                #print "key=",key," v=",doc['value']
                static_list_json[key]=eval(doc['value'])
            except:
                static_list_json[key]=None
                continue
        document_json["static_list_json"]=static_list_json

    return document_json
    
# clean up hdfs file    
def ml_clean_up_hdfs_file(hdfs_filename):             
    # remove hdfs://host:port
    if hdfs_filename.find("hdfs://")==0:
        idx=hdfs_filename.find("/", hdfs_filename.find(":",6)) # find path after :<port>
        hdfs_filename=hdfs_filename[idx:]
        
    try:
        hdfs.rmr(hdfs_filename)
        print "INFO: HDFS file",os.path.basename(hdfs_filename),"removed!"
    except IOError as e:
        #print "WARNING:({0}): {1}".format(e.errno, e.strerror)
        pass
    except:
        print "WARNING:", sys.exc_info()[0]    
                
# remove file in hdfs and replace with new one
def ml_overwrite_hdfs_file(filename, hdfs_filename):
    ml_clean_up_hdfs_file(hdfs_filename)
    
    try:
        ret=hdfs.put(filename,hdfs_filename)
        # no return code? print "ret=",ret
        print "INFO: HDFS file",os.path.basename(hdfs_filename),"uploaded!"
        ret=0
    except IOError as e:
        print "ERROR:({0}): {1}".format(e.errno, e.strerror)
    except:
        print "ERROR:", sys.exc_info()[0]   
    return ret

# check if file exists in hdfs 
def ml_ls_hdfs_file(hdfs_filename):
    # remove hdfs://host:port
    if hdfs_filename.find("hdfs://")==0:
        idx=hdfs_filename.find("/", hdfs_filename.find(":",6)) # find path after :<port>
        hdfs_filename=hdfs_filename[idx:]
        
    print "INFO: check ["+hdfs_filename+"] in hdfs "
    list=None
    try:
        list=hdfs.ls(hdfs_filename)
    except IOError as e:
        print "WARNING:({0}): {1}".format(e.errno, e.strerror)
    except:
        print "WARNING:", sys.exc_info()[0]   
        
    return list
    
# re-create a local file for write
def ml_recreate_file_4write(filename):
    f=None
    if os.path.exists(filename):
        try:
            os.remove(filename)
        except OSError, e:
            print ("ERROR: %s - %s." % (e.filename,e.strerror))
    try:        
        f=open(filename, 'w') 
    except OSError, e:
        print "ERROR:({0}): {1}".format(e.errno, e.strerror)
    except:
        print "ERROR:", sys.exc_info()[0]   
    
    return f

magic_dict = {
    "\x1f\x8b\x08": "gz",
    "\x42\x5a\x68": "bz2",
    "\x50\x4b\x03\x04": "zip"
    }

max_len = max(len(x) for x in magic_dict)

# check file type =======================
def file_type(filename):
    with open(filename) as f:
        file_start = f.read(max_len)
    for magic, filetype in magic_dict.items():
        if file_start.startswith(magic):
            return filetype
    return "no match"    
    
# find labels and last feature in libsvm =======================
# format : optionsl(key) label sparse_vector
def ml_get_lbl_dict_max_feat_from_libsvm(fname):
    label_dict = {}
    max_feat=0
    # TBD add stuff for zip here; compressed file is not support in Spark yet.
    if file_type(fname) == "gz":
        infs=gzip.open(fname, "r")
    else:
        infs=open(fname, "r")
    try:
        for line in infs:
            #splict libsvm line
            lst=line.strip().split(' ')
            if len(lst)>2:
                # get label
                lbl=lst[0].strip()
                # check if is a number, first item may be hash
                try:
                    lbl=int(lbl)
                    if not str(lbl) in label_dict:
                        label_dict[str(lbl)]=lbl
                except ValueError:
                    # get 2nd item
                    lbl=lst[1].strip()
                    try:
                        lbl=int(lbl)
                        if not str(lbl) in label_dict:
                            label_dict[str(lbl)]=lbl
                    except ValueError:
                        print "ERROR: libsvm format error"
                        break
                last_feat=lst[-1].split(':')[0].strip()
                if len(last_feat)>0 and int(last_feat)>max_feat:
                    max_feat=int(last_feat)
    except:
        print "ERROR: file io error."
    infs.close()
    return label_dict, max_feat

# save labels from libsvm line to dict 
def ml_save_libsvm_metadata(fname, row_id_str, key
        , ip_address, port, db_name, tb_name, username, password):
        
    # get label dict and max_feat
    label_dict, max_feat=ml_get_lbl_dict_max_feat_from_libsvm(fname)
    
    #print label_dict, max_feat
    
    #convert dict to array of json
    arr_labels=[]
    for kp in [(k,v) for k,v in label_dict.iteritems()] :
        arr_labels.append({kp[0]:kp[1]})
        
    # save to mongo
    jstr_dict=json.dumps(arr_labels)
    save_dict(row_id_str, key, jstr_dict
        , ip_address, port, db_name, tb_name, username, password)
    
    # save to local file
    # get path
    ldir=os.path.dirname(fname)
    ofname=os.path.join(ldir, 'libsvm_data_feat_count')
    #print "ofname=",ofname
    with open(ofname, "w+") as of:
        of.write(str(max_feat)+'\n')
    
    return 0
    
# EOLed save labels from libsvm line to dict 
def save_libsvm_label(fname, row_id_str, key
        , ip_address, port, db_name, tb_name, username, password):
    labels = {}
    with open(fname, "r") as infs:
        for line in infs:
            l=line.split(' ',1)[0].strip()
            if len(l) >0:
                labels[str(l)]=int(l)
        print "INFO: labels=",labels
    #convert to array of json
    arr_labels=[]
    for kp in [(k,v) for k,v in labels.iteritems()] :
        arr_labels.append({kp[0]:kp[1]})
    jstr_dict=json.dumps(arr_labels)
    #print jstr_dict
    # save to mongo
    save_dict(row_id_str, key, jstr_dict
        , ip_address, port, db_name, tb_name, username, password)

# save json string ===========================================================
def save_dict_t(row_id_str, key, jstr_dict, mongo_tuples):
    return save_dict(row_id_str, key, jstr_dict
        , mongo_tuples[0], mongo_tuples[1], mongo_tuples[2] 
        , mongo_tuples[3], mongo_tuples[4], mongo_tuples[5])
    
def save_dict(row_id_str, key, jstr_dict
        , ip_address, port, db_name, tb_name, username, password):
    # json for insert
    jobj={}
    jobj["rid"]=int(row_id_str)
    jobj["key"]=key
    jobj["value"]=json.loads(jstr_dict) # convert to json obj
    # convert to string
    jstr_insert=json.dumps(jobj)
    filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
    upsert_flag=True
    #print "jstr_insert=",jstr_insert
    ## write to mongoDB.myml.dataset_info, ignore doc with duplicated key
    ret=query_mongo.upsert_doc(ip_address, port, db_name, tb_name, username, password
            ,filter, jstr_insert, upsert_flag)
    print "INFO: save_dict id="+row_id_str+", key="+key+", rc=",ret
    return ret

# save json obj ===========================================================
def save_json_t(row_id_str, key, json_obj, mongo_tuples):
    return save_json(row_id_str, key, json_obj
        , mongo_tuples[0], mongo_tuples[1], mongo_tuples[2] 
        , mongo_tuples[3], mongo_tuples[4], mongo_tuples[5])

def save_json(row_id_str, key, json_obj
        , ip_address, port, db_name, tb_name, username, password):
    # json for insert
    jobj={}
    jobj["rid"]=int(row_id_str)
    jobj["key"]=key
    jobj["value"]=json_obj# convert to json obj
    # convert to string
    jstr_insert=json.dumps(jobj)
    filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
    upsert_flag=True
    #print "jstr_insert=",jstr_insert
    ## write to mongoDB.myml.dataset_info, ignore doc with duplicated key
    ret=query_mongo.upsert_doc(ip_address, port, db_name, tb_name, username, password
            ,filter, jstr_insert, upsert_flag)
    print "INFO: save_dict id="+row_id_str+", key="+key+", rc=",ret
    return ret
    

# get excluded_feat ================
def ml_get_excluded_feat(row_id_str, mongo_tuples):
    excluded_feat_cslist=None
    # get excluded feature list from mongo ========== ===
    key = "feature_excluded"
    jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
    jstr_proj='{"value":1}'
    # get from own id (not from parent dataset id)
    #print "jstr_filter=",jstr_filter,",jstr_proj=",jstr_proj
    doc=query_mongo.find_one_t(mongo_tuples, jstr_filter, jstr_proj)
    #print "feature_excluded=",doc
    if not doc is None and 'value' in doc:
        excluded_feat_cslist = ','.join(str(i) for i in doc['value'])
    return excluded_feat_cslist

    
# Create ROC graphs and JSON file for interactive graphs ==========================
def ml_create_roc_files(row_id_str, y_score, y_true ,testing_N_count, testing_P_count
        , out_dir, file_name_given): 

    ###########plot ROC figure#######
    try:
        fpr, tpr, thresholds = roc_curve(y_true , y_score, pos_label = 1)
        roc_auc = auc(fpr, tpr)
    except ValueError as e:
        print "ERROR: in ROC curve: ", e
    

    print "INFO: ROC_AUC = ", roc_auc
    total_count=len(fpr)
    mod_c=math.ceil(total_count/ROC_DOT_MAX_CNT)
    # mod_c ==1 for modular won't work. set to 2*
    if mod_c ==1:
        mod_c=2
    #print "INFO: len(fpr)=", len(fpr)
    print "INFO: ROC_DOT_MAX_CNT=", ROC_DOT_MAX_CNT
    print "INFO: mod count=", mod_c
    
    # create ROC data for graph ====================
    all_json=[]
    roc_json=[]
    acc_json=[]
    # ROC curve: add points when less than max count or 
    for idx,x in enumerate(fpr):
        if total_count<=ROC_DOT_MAX_CNT or idx % mod_c ==1:
            acc = (testing_P_count*tpr[idx] + testing_N_count*(1-fpr[idx]))/(testing_P_count + testing_N_count)
            roc_json.append([x,tpr[idx]]) # use array to reduce size
            acc_json.append([x,acc])
    # ROC data 
    plt_json={}
    plt_json["values"]=roc_json
    plt_json["key"]='ROC curve' #'ROC curve (area = %0.2f)' % roc_auc
    plt_json["color"]="#0080FF" # light blue
    all_json.append(plt_json)
    # accuray
    plt2_json={}
    plt2_json["values"]=acc_json
    plt2_json["key"]='Accuracy' 
    plt2_json["color"]="#FF4000" # red 
    all_json.append(plt2_json)
    # dot line
    #print "INFO: all_json=",all_json  

    ROC_jfile = os.path.join(out_dir,row_id_str+"_roc.json")
    print "INFO: ROC_jfile=",ROC_jfile
    if os.path.exists(ROC_jfile):
        try:
            os.remove(ROC_jfile)
        except OSError, e:
            print ("ERROR: %s - %s." % (e.ROC_jfile,e.strerror))

    try:
        with open(ROC_jfile,"w") as json_file:
            json.dump(all_json, json_file)
    except Exception as e:
        print "ERROR: ", e

    # plot figure here =============================== 
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(out_dir,file_name_given+"_ROC"+".png"))
    print "INFO: Figure created!"
    
    #### generate fpr tpr ACC threshold results file###
    ROC_file = os.path.join(out_dir,file_name_given+"_ROC_value.txt")
    if os.path.exists(ROC_file):
        try:
            os.remove(ROC_file)
        except OSError, e:
            print ("ERROR: %s - %s." % (e.ROC_file,e.strerror))
    
    for i in range(0, len(fpr)):
        ACC = (testing_P_count*tpr[i] + testing_N_count*(1-fpr[i]))/(testing_P_count + testing_N_count)
        with open(ROC_file, 'a') as f:
            f.write('%0.5f  ' % (fpr[i]))
            f.write('%0.5f  ' % (tpr[i]))
            f.write('%0.5f  ' % (thresholds[i]))
            f.write('%0.5f\n' % (ACC))
            
    return roc_auc

# get feature's raw string data ===================================
def feats2strs(str_feats, dic_hash_str):
    ret=""
    comma=""
    #print "str_feats=",str_feats
    # get string from dic_hash_str
    for f in str_feats.split(','):
        if f in dic_hash_str:
            ret = ret + comma + dic_hash_str[f]
            comma=","
        else:
            try:
                # may in number
                if int(f) in dic_hash_str:
                    ret = ret + comma + dic_hash_str[int(f)]
                else:
                    ret = ret + comma + "None"
            except:
                ret = ret + comma + "None"
            comma=","
            
    return ret

# output {"data":[{row1 json},{row2 json},...]} to a file ========================
def build_feat_list_t(rid, feat_file, dic_seq_hashes, dic_hash_str, coef_arr, ds_id, mongo_tuples, feat_sample_count_arr=None):
    return build_feat_list(rid, feat_file, dic_seq_hashes, dic_hash_str, coef_arr, ds_id
        , mongo_tuples[0], mongo_tuples[1], mongo_tuples[2] 
        , mongo_tuples[3], mongo_tuples[4], mongo_tuples[5] , feat_sample_count_arr)
        
def build_feat_list(rid, feat_file, dic_seq_hashes, dic_hash_str, coef_arr, ds_id
    , ip_address, port, db_name, tb_name, username, password, feat_sample_count_arr=None): 
    #print "I am in================================= ============="
    jstr_proj='{"value":1}'

    # coefficient list 
    if coef_arr is None:
        # get dic_hash_str ========================
        key = "coef_arr"
        jstr_filter='{"rid":'+rid+',"key":"'+key+'"}'
        jstr_proj='{"value":1}'
        # coef is bound to model
        #if ds_id != rid:
        #    jstr_filter='{"rid":'+ds_id+',"key":"'+key+'"}'
        doc=query_mongo.find_one(ip_address, port, db_name, tb_name, username, password, jstr_filter, jstr_proj)
        if doc:
            coef_arr = doc['value']
        else:
            coef_arr=None
            print "ERROR: coef_arr not found in mongo!"
            # get data from local file
            try:
                ldir=os.path.dirname(feat_file)
                fn=os.path.join(ldir,rid+"_coef_arr.pkl")
                print "INFO: get coef_arr from local", fn
                coef_arr=ml_pickle_load(fn)
                print "INFO: len(coef_arr)=", len(coef_arr)
            except:
                return None
            if coef_arr is None:
                return None
    #print "coef_arr=",coef_arr
    
    # hash to raw string dict
    if dic_hash_str is None:
        # get dic_hash_str ========================
        key = "dic_hash_str"
        jstr_filter='{"rid":'+rid+',"key":"'+key+'"}'
        jstr_proj='{"value":1}'
        # get parent dataset's data
        if ds_id != rid:
            jstr_filter='{"rid":'+ds_id+',"key":"'+key+'"}'
        doc=query_mongo.find_one(ip_address, port, db_name, tb_name, username, password, jstr_filter, jstr_proj)
        if doc:
            dic_hash_str = doc['value']
        else:
            dic_hash_str=None
            print "WARNING: dic_hash_str not found in mongo!"
            # get data from local file
            try:
                ldir=os.path.dirname(feat_file)
                fn=os.path.join(ldir,rid+"_dic_hash_str.pkl")
                print "INFO: get dic_hash_str from local", fn
                dic_hash_str=ml_pickle_load(fn)
                print "INFO: len(dic_hash_str)=", len(dic_hash_str)
            except:
                return None
            if dic_hash_str is None:
                return None
    #print "dic_hash_str=",dic_hash_str
      
    # feat id/sequence to hash dict
    if dic_seq_hashes is None:
        # get dic_seq_hashes ========================
        key = "dic_seq_hashes"
        jstr_filter='{"rid":'+rid+',"key":"'+key+'"}'
        # get parent dataset's data
        if ds_id != rid:
            jstr_filter='{"rid":'+ds_id+',"key":"'+key+'"}'
        doc=query_mongo.find_one(ip_address, port, db_name, tb_name, username, password, jstr_filter, jstr_proj)
        if doc:
            dic_seq_hashes = doc['value']
        else:
            dic_seq_hashes=None 
            print "WARNING: dic_seq_hashes not found in mongo!"
            # get data from local file
            try:
                ldir=os.path.dirname(feat_file)
                fn=os.path.join(ldir,rid+"_dic_seq_hashes.pkl")
                print "INFO: get coef_arr from local", fn
                dic_seq_hashes=ml_pickle_load(fn)
                print "INFO: len(dic_seq_hashes)=", len(dic_seq_hashes)
            except:
                return None
            if dic_seq_hashes is None:
                return None
    # sample count of feat id
    if feat_sample_count_arr is None:
        # get feat_sample_count_arr ========================
        key = "feat_sample_count_arr"
        jstr_filter='{"rid":'+rid+',"key":"'+key+'"}'
        jstr_proj='{"value":1}'
        # get parent dataset's data
        if ds_id != rid:
            jstr_filter='{"rid":'+ds_id+',"key":"'+key+'"}'    
        doc=query_mongo.find_one(ip_address, port, db_name, tb_name, username, password, jstr_filter, jstr_proj)
        if doc:
            feat_sample_count_arr = doc['value']
        else:
            feat_sample_count_arr=None 
            print "WARNING: feat_sample_count_arr not found in mongo!"
            # get data from local file
            try:
                ldir=os.path.dirname(feat_file)
                fn=os.path.join(ldir,rid+"_feat_sample_count_arr.pkl")
                print "INFO: get feat_sample_count_arr from local", fn
                feat_sample_count_arr=ml_pickle_load(fn)
                print "INFO: len(feat_sample_count_arr)=", len(feat_sample_count_arr)
            except:
                return None
            if feat_sample_count_arr is None:
                return None
    arr=[]
    all_json={}
    ret_json={}
    if not dic_seq_hashes is None and not dic_hash_str is None:
        for k,v in dic_seq_hashes.iteritems():
            j={}
            j["fid"]=k
            j["n_gram"]=v
            strs=feats2strs(v, dic_hash_str)
            j["raw_string"]=strs
            if not coef_arr is None and len(coef_arr)>int(k):
                j["coef"]=coef_arr[int(k)-1] # 0 based idx in array
            else:
                j["coef"]=0
            if not feat_sample_count_arr is None and len(feat_sample_count_arr)>int(k):
                j["feat_sample_count"]=feat_sample_count_arr[int(k)-1] # 0 based idx in array
            else:
                j["feat_sample_count"]=0
            arr.append(j)
            ret_json[k]=(j["coef"],strs,j["feat_sample_count"])
      
        all_json["data"]=arr
        # dump all_json to a file
        if os.path.exists(feat_file):
            try:
                os.remove(feat_file)
            except OSError, e:
                print ("ERROR: %s - %s." % (e.feat_file,e.strerror))

        try:
            with open(feat_file,"w") as json_file:
                json.dump(all_json, json_file)
        except Exception as e:
            print "ERROR: ",e
    
    return ret_json # dict of {fid:(coef,raw_string,feat_sample_count)}

# for IN or libsvm only
def build_feat_coef_raw_list_t(rid, feat_filename, coef_arr, ds_id, mongo_tuples, feat_sample_count_arr=None):
    jfeat_coef_dict={}    
    # IN dynamic only
    jret=None # TBD
        
    if not jret is None:
        # jret is a dict of {"data":[ dict of {fid,coef,raw_string,n_gram} ]
        try: # new format with feat_sample_count
            # TBD to avoid hard coded name here
            for i, v in enumerate( (d['fid'],(d['coef'],d['raw_string'],d['feat_sample_count'])) for d in jret["data"]):
                jfeat_coef_dict[v[0]]=v[1]
        except: # for old version
            for i, v in enumerate( (d['fid'],(d['coef'],d['raw_string'])) for d in jret["data"]):
                jfeat_coef_dict[v[0]]=v[1]            
        #print "HiHi jfeat_coef_dict=",jfeat_coef_dict
        return jfeat_coef_dict
        
     
    jret=None # TBD
    if not jret is None:
        for i, v in enumerate( (d['fid'],(d['coef'],d['raw_string'])) for d in jret["data"]):
            jfeat_coef_dict[v[0]]=v[1]           
        return jfeat_coef_dict
 
    # libsvm only
    jret=build_feat_list_libsvm(rid, feat_filename, coef_arr)
    if not jret is None:
        #print "jret=",jret
        for i, v in enumerate( (d['fid'],(d['coef'],d['raw_string'])) for d in jret["data"]):
            jfeat_coef_dict[str(v[0])]=v[1]           
        return jfeat_coef_dict


        
    return jfeat_coef_dict

    
# output {"data":[{row1 json},{row2 json},...]} to a file; for dataset type "libsvm" ONLY ==================
def build_feat_list_libsvm(rid, feat_file, coef_arr): 

    #print "INFO: in build_feat_list_libsvm() "
    arr=[]
    all_json={}

    for idx,v in enumerate(coef_arr):
        j={}
        j["fid"]=idx+1
        j["n_gram"]=""
        j["raw_string"]=""
        j["coef"]=v
        arr.append(j)
  
    all_json["data"]=arr
    # dump to a file
    if os.path.exists(feat_file):
        try:
            os.remove(feat_file)
        except OSError, e:
            print ("ERROR: %s - %s." % (e.feat_file,e.strerror))

    try:
        with open(feat_file,"w") as json_file:
            json.dump(all_json, json_file)
    except Exception as e:
        print "ERROR: ",e
    
    return all_json

  
# calculate_hypothesis for prediction =================
def calculate_hypothesis(curr_dic, col_num, coef_arr, intercept, model_classname):
    # convert to numpy array
    curr_arr= np.asarray(generate_vector_from_dic(curr_dic, col_num))
    if intercept is None:
        intercept=0.0
        print "WARNING: intercept is None! at calculate_hypothesis()"
        
    # multiple class
    coef_len=len(coef_arr)
    curr_len=len(curr_arr)
    hypothesis_val=None
    if curr_len != coef_len  and coef_len % curr_len==0:
        print "INFO: multi-class classification found!"
        class_cnt=coef_len/curr_len
        hypothesis_val=0
        class_idx=0
        
        # class 0 is the pivot class, if all val is negative, then it is class 0
        for i in range(class_cnt):
            # no intercept in multiple class
            v=np.dot(curr_arr,coef_arr[i*curr_len:i*curr_len+curr_len])
            if v > hypothesis_val:
                hypothesis_val=v
                class_idx=i+1
        print "INFO: predicted class idx=",class_idx

    else:
        # theta' * X + theta_0
        hypothesis_val=np.dot(curr_arr,coef_arr)+intercept

    # LogisticRegression need to be sigmoid()-ed ====== ====
    if model_classname and "logistic" in model_classname.lower():
        hypothesis_val=sigmoid(hypothesis_val) 

    return hypothesis_val
    
# calculate_hypothesis for prediction; sparse array version =================
def calculate_hypothesis_arr(v_array, coef_arr, intercept, model_classname):
    hypothesis_val=None
    if not coef_arr is None and not v_array is None:
        hypothesis_val=np.dot(v_array,coef_arr) + intercept
        if model_classname and "logistic" in model_classname.lower():
            hypothesis_val=sigmoid(hypothesis_val) 
    return hypothesis_val
    
# for logistic regression to calculate hypothesis output =================
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# generate the vector from dict; shift feature to zero-based; set all feature value to 1   =================
def generate_vector_from_dic(curr_dic, col_num):
    vector_test = [0] * col_num
    for key in curr_dic:
        col = int(key)
        #print "col=",col
        vector_test[col - 1] = curr_dic[key]
    
    return vector_test

# generate the sparse matrix from curr_dic; set all feature value to 1  =======================
def generate_matrix_from_dic(curr_dic, col_num):
    row_testing = []
    col_testing = []
    features_testing = []
    
    for key in curr_dic:
        if (int(key)-1) > col_num:
            pass    #omit the new features
        else:
            # binary: all set to one?
            #feature_val = min(1, curr_dic[key])
            # use value
            feature_val =curr_dic[key]
            row_testing.append(0)
            col_testing.append(int(key) - 1)
            features_testing.append(feature_val)
          
    features_testing = np.array(features_testing)
    row_testing = np.array(row_testing)
    col_testing = np.array(col_testing)
   
    row_num = 1
    sparse_test = csr_matrix((features_testing,(row_testing,col_testing)), shape=(row_num,col_num))
    return sparse_test
    
# return (fscore,precision,recall,accuracy)
def calculate_fscore(true_label_arr, predict_label_arr):
    # calculate fscore  ==========
    testing_sample_number=len(true_label_arr)
    tp=0
    fp=0
    fn=0
    tn=0
    for idx, v in enumerate(true_label_arr):
        p=predict_label_arr[idx]
        if v==1:
            if p==1: #v == 1 and p==1
                tp=tp+1
            else:    #v == 1 and p==0
                fn=fn+1
        else:
            if p==1: #v == 0 and p==1
                fp=fp+1
            else:    #v == 0 and p==0
                tn=tn+1       
    #print "tp=",tp,",fp=",fp,",fn=",fn,",tn=",tn
    if tp+fp == 0 :
        precision=0
        print "WARNING: true positive + false positive is zero"
    else:
        precision=float(tp)/(tp+fp)  
        
    if tp+fn == 0 :
        recall=0
        print "WARNING: true positive + false negative is zero"
    else:
        recall=float(tp)/(tp+fn)
    
    if tp+fp==0 or tp+fn==0 or tn+fp==0 or tn+fn==0:
        phi=0
        print "WARNING: tp+fp==0 or tp+fn==0 or tn+fp==0 or tn+fn==0"
    else:
        phi=(float(tp*tn) - fp*fn)/math.sqrt(float(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        
    if precision+recall==0:
        fscore=0
    else:
        fscore=2*((precision*recall)/(precision+recall))
    #print "precision=",precision,",recall=",recall
    
    if testing_sample_number>0:
        accuracy=(tp+tn)/(float(testing_sample_number))
    else:
        accuracy=0

    #print "fscore=",fscore,",accuracy=",accuracy
    return {"fscore":fscore,"precision":precision,"recall":recall,"accuracy":accuracy, "phi":phi,"tp":tp,"tn":tn,"fp":fp,"fn":fn}

    # save data for sklearn pickle package
def ml_pickle_save(obj, filename):
    dirname = os.path.dirname(filename)
    if (dirname != '') and (os.path.isdir(dirname) == False):
        os.makedirs(dirname)
        
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, -1)
        f.close()
    
# load data for sklearn pickle package
def ml_pickle_load(filename):
    ret=None
    with open(filename, 'rb') as f:
        ret = pickle.load(f)
        f.close()
    return ret


#######################
####plot functions ####
#######################
# also generate data files for 3D 
def ml_plot_kmeans_dot_graph_save_file(mtx_feat, mtx_label, mtx_center, n_clusters, figsize=(10,7), filename=None
        , title='KMeans', filename_3d=None): 
    PLOT_GRID_TH=111 # 1x1 grid, 1st subplot
    # 
    cmap = get_cmap(n_clusters+1)
    colors = []
    for i in range(n_clusters):
        col = cmap(i)
        colors.append(col)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(PLOT_GRID_TH)

    print "INFO: mtx_feat.shape=",mtx_feat.shape ,", mtx_feat t=",type(mtx_feat) #, ",mtx_feat=",mtx_feat
    print "INFO: mtx_label.shape=",mtx_label.shape ,", mtx_label t=",type(mtx_label)
    print "INFO: mtx_center.shape=",mtx_center.shape ,", mtx_center t=",type(mtx_center)
    # plot for each color
    g_data={}
    id_arr=[]
    x_arr=[]
    y_arr=[]
    z_arr=[]
    style_arr=[]
    count=1
    
    cluster_color=zip(range(n_clusters), colors)
    print "INFO:  **zip(range(n_clusters), colors)=", cluster_color
    for k, col in cluster_color: #zip(range(n_clusters), colors):
        my_members = mtx_label == k
        
        print "INFO: mtx_feat[my_members, 0].shape=",mtx_feat[my_members, 0].shape #,", mtx_feat[my_members, 0]=",mtx_feat[my_members, 0]
        cluster_center = mtx_center[k] 
        print "INFO: k=",k,",cluster_center[0]=",cluster_center[0],", cluster_center[1]=",cluster_center[1]
        g={}

        # add dots for 3d  here
        x_list=mtx_feat[my_members, 0].tolist()
        x_len=len(x_list)
        x_arr=x_arr+x_list
        y_arr=y_arr+mtx_feat[my_members, 1].tolist()
        z_arr=z_arr+mtx_feat[my_members, 2].tolist()
        style_arr=style_arr+([k]*x_len) # center share same style
        # fill out id_arr with sequential number?
        for i in range(count,x_len+count): 
            id_arr.append(i)
        count=count+x_len
        
        # plot data points
        ax.plot(mtx_feat[my_members, 0], mtx_feat[my_members, 1], 'w',
                markerfacecolor=col, marker='.' , markersize=6)
    #add centers
    for c, col in cluster_color: #range(n_clusters):
        x_arr.append(mtx_center[c][0])
        y_arr.append(mtx_center[c][1])
        z_arr.append(mtx_center[c][2])
        id_arr.append(c*-1)
        style_arr.append(n_clusters)
        # plot centre dots after data dots
        ax.plot(mtx_center[c][0], mtx_center[c][1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=10)

    g_data["x"]=x_arr
    g_data["y"]=y_arr
    g_data["z"]=z_arr
    g_data["id"]=id_arr
    g_data["style"]=style_arr
    
    ax.set_title(title)
    #ax.set_xticks(())
    #### annotate cluster index ###
    for k in range(n_clusters):
        cluster_center = mtx_center[k]
        annotate_str = str(k)
        ax.annotate(annotate_str, xy=(cluster_center[0], cluster_center[1]), 
                rotation=15, 
                verticalalignment='bottom', 
                horizontalalignment='left')
    if filename:
        fig.savefig(filename)
        print "INFO: 2D Figure saved=" ,filename


    if filename_3d:
        #### generate 3d data file###
        if os.path.exists(filename_3d):
            try:
                os.remove(filename_3d)
            except OSError, e:
                print ("ERROR: %s - %s." % (e.filename_3d,e.strerror))
        with open(filename_3d, 'w') as f:
            json.dump(g_data,f)
            print "INFO: filename_3d=",filename_3d

    #print "========== g_data =",g_data
    return fig    
    
#Returns a function that maps each index in 0, 1, ... N-1 to a distinct RGB color.  ===========
def get_cmap(N):
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color
    
#  ml_plot_kmeans_histogram_subfigures ==========================
def ml_plot_kmeans_histogram_subfigures(labels_true, labels_predicted, n_clusters, names = {}, num_names_to_show = 3
            , plot_col_num = 4, figsize=(16,12), reverse = False, normalize = False, folder = 'results', rid=""):
        
    K = int(math.ceil(n_clusters*1.0/plot_col_num))
    fig, ax_all = plt.subplots(K, plot_col_num, sharex=True, figsize=figsize)
    ax_all = ax_all.flatten()
    #true_label_num = len(np.unique(labels_true))
    true_label_num = int(max(labels_true) + 1)
    
    if num_names_to_show > true_label_num:
        num_names_to_show = true_label_num
        print "WARNING: n_clusters < num_names_to_show, so set num_names_to_show = ", num_names_to_show
    
    if normalize == True:
        sample_num_vec = []
        for k in range(true_label_num):
            num = len(labels_true[labels_true == k])
            if num > 0:
                sample_num_vec.append(len(labels_true[labels_true == k]))
            else:
                sample_num_vec.append(0.0001)
        #print "sample_num_vec:", sample_num_vec
        
    for k in range(int(n_clusters)):
            
        n, bins = np.histogram(labels_true[labels_predicted == k], bins=range( true_label_num+1) )
        if np.sum(n) > 0:
            if normalize == True:
                nor = 1.0 * n / np.array(sample_num_vec) * sum(sample_num_vec)       
                #print "normalized n:", nor
                p = 1.0 * nor / np.sum(nor)
            else:
                p = 1.0 * n / np.sum(n)
        else:
            p = 1.0 * n

        for i in range(len(p)):
            if p[i] > 1:
                p[i] = 0

        pp = np.expand_dims(p, axis=0)
        if k == 0:
            all_p = pp
        else:
            all_p = np.concatenate((all_p, pp), axis=0)
        
        
        shownames = np.arange(len(p))
        shownames = map(str, shownames)
        
        index_according_to_order = [i[0] for i in sorted(enumerate(p), key=lambda x:x[1], reverse=True)]
        for i in range(num_names_to_show):
            idx = index_according_to_order[i]
            if reverse == False:
                if (len(names) > 0):
                    shownames[idx] = str(idx) + ":" + names[idx]
                else:
                    pass
            else:   #reverse == True
                shownames[idx] = str(idx) + ":" + str(idx)
                    
        
        if reverse == False:
            title_str = 'cluster %d (' % k + '%d)' % sum(n) + ', H:%.2f' % entropy(p)
        else:
            if np.sum(p) > 0:
                title_str = '%d: '%k + '%s (' % names[k] + '%d)' % sum(n) + ', H:%.2f' % entropy(p)
            else:
                title_str = '%d: '%k + '%s (' % names[k] + '%d)' % sum(n)
        
        plot_one_multinomial(p, bins, names = shownames, title = title_str, label = '1', color='b', ax=ax_all[k])
        #ax_all[k].legend(loc='upper right')
        plt.tight_layout()
    if (reverse == False) and (normalize == False):
        fig.savefig(os.path.join(folder ,rid+'_histogram_true_labels.png'))
        print "INFO: Figure saved: " ,os.path.join(folder ,rid+'_histogram_true_labels.png')
    elif (reverse == False) and (normalize == True):
        fig.savefig(os.path.join(folder ,rid+'_histogram_true_labels_normalized.png'))
        print "INFO: Figure saved: " , os.path.join(folder ,rid+'_histogram_true_labels_normalized.png')
    else:
        fig.savefig(os.path.join(folder ,rid+'_histogram_predicted_labels.png'))
        print "INFO: Figure saved: " ,os.path.join(folder ,rid+'_histogram_predicted_labels.png')
    return fig, all_p

# plot_one_multinomial ==========================
def plot_one_multinomial(vals, bins, names, title='probability distribution', label='', ax=None, color='b'):
    
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
    #ind = np.arange(len(vals))
    rects = ax.bar(bins[0:-1], vals, color=color, align='center', width=0.9)
    ax.set_xticks(bins[0:-1])
    ax.set_xlim((-1,len(vals)))
    ax.set_ylim((0,max(vals)+0.1))
    
    ax.set_title(title)
    
    idx_sorted = np.argsort(vals)
    for i in range(len(vals)):
        ii = idx_sorted[-i]
        ax.annotate(names[ii], xy=(ii, vals[ii]), 
                    rotation=15, 
                    verticalalignment='bottom', 
                    horizontalalignment='left')

# convert dict to libsvm format: {"1":1}
def dict2libsvm_str(label, dict_feat, set_all_one="Y"):
    ret=str(label)
    sorted_dict= sorted(dict_feat.items(),key=(lambda x:int(x[0])))
    for i in sorted_dict:
        if set_all_one=="Y":
            ret="%s%s%s%s" %(ret," ",str(i[0]),":1")
        else:
            ret="%s%s%s%s%s" %(ret," ",str(i[0]),":",str(i[1]))
        
    return ret

# convert libsvm string to tuple (label, dict={"mm":nn})
def libsvm_str2tuple(libsvm_str):
    if libsvm_str is None:
        return (None,None)
    feat={}
    arr=libsvm_str.split(' ')
    if len(arr)>1:
        label=arr[0]
        for i in arr[1:]:
            (f,v)=i.split(':')
            feat[f]=v
        return (label, feat)
    else:  
        return (libsvm_str,None)   
    
# invoke pca_local.sh to transform a libsvm string by Scala
def ml_mllib_pca_local_transform(rid,libsvm_str,pca_k,ds_id):
    #print "INFO: in mllib_pca_local_transform, id=",rid
    rid=str(rid)
    pca_k=str(pca_k)
    ds_id=str(ds_id)
    # invoke Scala program to load PCA model and transform
    ret=subprocess.call([config.get('app', 'TASK_EXE'),      #bash
                config.get('app', 'PCA_LOCAL_SCRIPT'),      #pca_local.sh
                rid,
                pca_k, 
                ds_id,
                libsvm_str,
    ])
    
    # set output filename
    out_file=os.path.join(config.get('app', 'TRAIN_DES_DIR'), rid, rid+"pca_local.txt")

    # open to read and return here
    f = open(out_file, 'r')
    new_libsvm_str = f.readline()
    #print "transformed=", line
    return new_libsvm_str

# prepare output local and hdfs dir; clean up old model files    
def ml_prepare_output_dirs(row_id_str, local_out_dir,model_data_folder,model_fname):
    # create folders
    if not os.path.exists(local_out_dir):
        os.makedirs(local_out_dir)
    if not os.path.exists(model_data_folder):
        os.makedirs(model_data_folder)
    #clean-up model files   
    if os.path.exists(model_fname):
        try:
            for fl in glob.glob(model_fname+"*"):
                os.remove(fl)
        except OSError, e:
            print ("Error: %s - %s." % (e.model_fname,e.strerror))

# return Spark context
def ml_get_spark_context(sp_master, p_compress, p_maxResultSize, p_exe_memory, p_core_max, sp_jobname="NoName"
        , zip_file_path=None, p_driver_memory="2g"):   
    from pyspark import SparkContext
    SparkContext.setSystemProperty('spark.rdd.compress', p_compress)
    SparkContext.setSystemProperty('spark.driver.maxResultSize', p_maxResultSize)
    SparkContext.setSystemProperty('spark.executor.memory', p_exe_memory)
    SparkContext.setSystemProperty('spark.cores.max', p_core_max)
    #SparkContext.setSystemProperty('spark.ui.showConsoleProgress', 'false')
    SparkContext.setSystemProperty('spark.ui.consoleProgress.update.interval', '60000')
    #SparkContext.setSystemProperty('spark.driver.memory', p_driver_memory)

    if not zip_file_path is None:
        sc = SparkContext(sp_master, sp_jobname, pyFiles=zip_file_path)
    else:
        sc = SparkContext(sp_master, sp_jobname)
    
    return sc    

# return Spark session
def ml_get_spark_session(sp_master, p_compress, p_maxResultSize, p_exe_memory, p_core_max, sp_jobname="NoName"
        , zip_file_path=None, p_driver_memory="2g"):   
    from pyspark.sql import SparkSession
    spark=SparkSession.builder \
        .master(sp_master) \
        .appName(sp_jobname) \
        .config('spark.rdd.compress', p_compress) \
        .config('spark.driver.maxResultSize', p_maxResultSize) \
        .config('spark.executor.memory', p_exe_memory) \
        .config('spark.cores.max', p_core_max) \
        .config('spark.ui.consoleProgress.update.interval', '60000') \
        .getOrCreate()

    if not zip_file_path is None:
        spark.sparkContext.addPyFile(zip_file_path)

    return spark    
    
    
# build false prediction table; for sklearn linear model only
def ml_build_false_pred(X_test_sparse,coef,intercept,labels_test,labels_pred,test_hash_list,model_name
        ,jfeat_coef_dict, false_pred_fname, row_id_str=None, ds_id=None
        , feat_sample_count_arr=None, mongo_tuples=None):
    # get False prediction & score graph data; format "tlabel":,"plabel":,"score":,"feat":{"<fid>":["<coef>","<raw_string>"]} =================
    false_pred_arr=[]
    score_arr_0=[]
    score_arr_1=[]
    max_score=0
    min_score=0
    arr=X_test_sparse.toarray()
    has_sample_count=False
    #print "arr=",arr
    #print "jfeat_coef_dict=",jfeat_coef_dict
    
    # sample count of feat id
    if feat_sample_count_arr is None and not mongo_tuples is None and len(mongo_tuples)>0 \
        and not row_id_str is None:
        # get feat_sample_count_arr ========================
        key = "feat_sample_count_arr"
        jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
        jstr_proj='{"value":1}'
        # get parent dataset's data
        if ds_id != row_id_str:
            jstr_filter='{"rid":'+ds_id+',"key":"'+key+'"}'    
        doc=query_mongo.find_one_t(mongo_tuples, jstr_filter, jstr_proj)
        if doc:
            feat_sample_count_arr = doc['value']
            if feat_sample_count_arr and len(feat_sample_count_arr)>0:
                has_sample_count=True
        else:
            feat_sample_count_arr=None 
            print "WARNING: feat_sample_count_arr not found!"

    # for each same, calculate score etc
    for tl_idx,tlbl in enumerate(labels_test):
        false_pred_j={}
        #pscore= np.dot(coef,arr[tl_idx,:]) + intercept
        # calculate prediction score
        pscore= calculate_hypothesis_arr(arr[tl_idx,:], coef, intercept, model_name)
        #print "pscore=",pscore, ", tlabel=",tlbl,", labels_pred=",labels_pred[tl_idx]
        
        # for false prediction
        if int(labels_pred[tl_idx]) != int(tlbl):
            false_pred_j["tlabel"]=tlbl
            false_pred_j["plabel"]=labels_pred[tl_idx]
            false_pred_j["score"]= pscore
            false_pred_j["hash"]= test_hash_list[tl_idx]

            #print "pscore=",pscore, ", tlabel=",tlbl,", labels_pred=",labels_pred[tl_idx]
            feat={}
            #print "arr=",arr
            #print "tl_idx=",tl_idx
            
            # get info for each feature
            for fid, v in enumerate(arr[tl_idx,:]):
                farr=[]
                if v > 0 and len(jfeat_coef_dict)>0:
                    farr=[jfeat_coef_dict[str(fid+1)][0],jfeat_coef_dict[str(fid+1)][1]]
                    if has_sample_count:
                        farr.append(feat_sample_count_arr[fid])
                    feat[str(fid+1)]=farr
            false_pred_j["feat"]=feat
            #print "false_pred_j=",false_pred_j
            false_pred_arr.append(false_pred_j)
            #print "score=", v, labels_pred[tl_idx], clf.predict(arr[tl_idx,:]), np.dot(coef,arr[tl_idx,:]) + intercept
        # for score graph, get a list of score for each true label
        if int(tlbl)==0:
            score_arr_0.append(pscore)
        else:
            score_arr_1.append(pscore)
        # save max,min score
        if pscore>max_score:
            max_score=pscore
        elif pscore<min_score:
            min_score=pscore

    # save false prediction to local file
    #false_pred_fname=os.path.join(local_out_dir,row_id_str+"_false_pred.json")
    if not false_pred_fname is None:
        with open (false_pred_fname,"w")as fp:
            fp.write(json.dumps(false_pred_arr))
    
    return (score_arr_0,score_arr_1,max_score,min_score)
            
    # create score histogram
    #ml_build_pred_score_graph(score_arr_0,score_arr_1,model_name, score_graph_fname)        
            
# create pred histogram graph for testing dataset
def ml_build_pred_score_graph(score_arr_0,score_arr_1,model_name, score_graph_fname,max_score,min_score):

    
    half_bin_count=BAR_BIN_COUNT/2
    if "logistic" in model_name.lower():
        boundary_pt=0.5
        bar_width=1.0/BAR_BIN_COUNT
        bar_width=float("%.5f" % bar_width )
        n_0, bins_0, patches_0 = plt.hist(score_arr_0, bins=BAR_BIN_COUNT, range=(boundary_pt-(bar_width*half_bin_count),boundary_pt+(bar_width*half_bin_count)))
        n_1, bins_1, patches_1 = plt.hist(score_arr_1, bins=BAR_BIN_COUNT, range=(boundary_pt-(bar_width*half_bin_count),boundary_pt+(bar_width*half_bin_count)))
    else:
        boundary_pt=0.0
        bar_width=(max_score-min_score)/BAR_BIN_COUNT
        bar_width=float("%.5f" % bar_width )
        # create score graph to local file (use pylot to build histogram)
        #print "bar_width=",bar_width
        n_0, bins_0, patches_0 = plt.hist(score_arr_0, bins=BAR_BIN_COUNT, range=(bar_width*(-1.0*half_bin_count),bar_width*half_bin_count))
        #print "n_0=",n_0,"bins_0=",bins_0,"patches_0=",patches_0
        n_1, bins_1, patches_1 = plt.hist(score_arr_1, bins=BAR_BIN_COUNT, range=(bar_width*(-1.0*half_bin_count),bar_width*half_bin_count))
        #print "n_1=",n_1'"bins_1=",bins_1'"patches_1=",patches_1
    
    # build to format [{"values": [[x,y],[x1,y1],...],  "type": "bar","key": "<any string>", "yAxis": 1 }, {<bar2>},{<boundary line>} ]
    bar1_v=[]
    bar0_v=[]
    max_n=0
    for idx,b0 in enumerate(n_0):
        b1=n_1[idx]
        x=bins_0[idx]
        if x>=boundary_pt: # skip x==0
            x=bins_0[idx+1]
        bar0_v.append([x,b0])
        bar1_v.append([x,b1])
        if b0>max_n:
            max_n=b0
        if b1>max_n:
            max_n=b1
    bar0={"values":bar0_v,"type":"bar","key":"clean","yAxis":1}    
    bar1={"values":bar1_v,"type":"bar","key":"dirty","yAxis":1}    
    bline={"values":[[boundary_pt,max_n],[boundary_pt,0.0]],"type":"line","key":"boundary","yAxis":1}   
    score_graph_data=[bar0,bar1,bline]   
    if not score_graph_fname is None:
        with open (score_graph_fname,"w")as fp:
            fp.write(json.dumps(score_graph_data))

# save coef and intercept to MongoDB ==========
def ml_save_coef_build_feat_coef(row_id_str, mongo_tuples, coef, intercept, feat_filename, ds_id):    
    key="coef_arr"
    coef_arr=coef.tolist()
    ret=save_json_t(row_id_str, key, coef_arr, mongo_tuples)
    # save intercept to mongo ===
    key="coef_intercept"
    ret=save_json_t(row_id_str, key, intercept, mongo_tuples)
    # create feature list + coef file =============================================== ============
    jret=build_feat_list_t(row_id_str, feat_filename, None, None, coef_arr, ds_id, mongo_tuples)

    # special featuring for IN or libsvm
    if jret is None:
        jret=build_feat_coef_raw_list_t(row_id_str, feat_filename, coef_arr, ds_id, mongo_tuples)
    if jret is None:
        print "WARNING: Cannot create sample list for testing dataset. "

    #jfeat_coef_dict=jret
    return jret

# plot predict figures for testing dataset
def ml_plot_predict_figures(pred_label_list, true_label_list, labels_list, label_dic, testing_sample_count
    , pred_xlabel, pred_fname, true_xlabel, true_fname):  
    
    ### generate sample numbers of each family in testing data ===============
    test_cnt_dic = {}
    for key in label_dic:
        test_cnt_dic[key] = 0
    for i in range (0, testing_sample_count):
        for key in label_dic:
            if true_label_list[i] == key:
                test_cnt_dic[key] = test_cnt_dic[key] + 1
    print "INFO: Number of samples in each label is=", test_cnt_dic

    # generate labels_list
    if labels_list is None or len(labels_list)==0:
        labels_list=[]
        for key in sorted(label_dic):
            labels_list.append(label_dic[key])
    
    ### reorder labels so that labels are ordered according to the true label of the data
    len_pred = len(pred_label_list)
    wide_len = int(math.ceil(math.sqrt(len_pred)))
    #print "wide_len=",wide_len
    
    #pred_list = labels_pred.tolist()
    #test_list = true_label_list.tolist()
    labels_true_pred = zip(true_label_list, pred_label_list)
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
    #plt.xlabel('Prediction (Single Run)')
    #plt.savefig(local_out_dir+file_name_given+"_1"+".png")
    plt.xlabel(pred_xlabel)
    plt.savefig(pred_fname)
    
    
    fig, ax = plt.subplots()
    cax = ax.imshow(mtx_testing, interpolation='nearest', cmap=plt.cm.jet)
    #ax.set_title('Gaussian noise with vertical colorbar')
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=tic)
    cbar.ax.set_yticklabels(labels_str)# vertically oriented colorbar
    cbar.ax.invert_yaxis()
    #plt.xlabel('True Labels (Single Run)')
    #plt.savefig(local_out_dir+file_name_given+"_2"+".png")
    plt.xlabel(true_xlabel)
    plt.savefig(true_fname)

    plt.show()
    return test_cnt_dic
    
# get label from mongo ============
def ml_get_label_dict(row_id_str, mongo_tuples, ds_id): 
    # format: [{"dirty":1}, {"clean":0}]
    key = "dic_name_label"
    jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
    jstr_proj='{"value":1}'
    
    # get parent dataset's data
    if ds_id != row_id_str:
        jstr_filter='{"rid":'+ds_id+',"key":"'+key+'"}'
    
    doc=query_mongo.find_one_t(mongo_tuples, jstr_filter, jstr_proj)
    if not doc is None and 'value' in doc:
        dict_list = doc['value']
    else:
        dict_list=None
    label_dict = {}
    #print "dict_list=",dict_list
    for i in range(0, len(dict_list)):
        for key in dict_list[i]:
            label_dict[dict_list[i][key]] = key.encode('UTF8')
    return label_dict    

# get n_components for pca variance array by threshold ============
def ml_get_n_components(var_arr,threshold):    
    sum_ratio = 0
    # get ratio array and find n_components 
    for n_components,val in enumerate(var_arr):
        sum_ratio=sum_ratio+val
        if sum_ratio >= threshold:
            break
    # index is zero based need to add 1
    return n_components+1    
if __name__ == '__main__':
    __description__ = "utilties for ml"
    main()
