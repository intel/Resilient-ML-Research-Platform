#!/usr/bin/python
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
import operator, json
from bson import json_util

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
####newly added###
from collections import OrderedDict

#####import for django database####
sys.path.append('./db')
import exec_sqlite
import query_mongo

####global constant
CONF_FILE='../../app.config' # at the base dir of the web
config=ConfigParser.ConfigParser()
config.read(CONF_FILE)

def main():
    
    parser = ArgumentParser(description=__description__)
    parser.add_argument("-n", "--number", type=str, metavar="number to show in list", help="number to show in list", required=False)
    parser.add_argument("-f", "--firm", type=str, metavar="FIRM score file", help="FIRM score file", required=False)
    parser.add_argument("-it", "--it", type=str, metavar="IT score file", help="IT score file", required=False)
    parser.add_argument("-pb", "--prob", type=str, metavar="Prob score file", help="Prob score file", required=False)
    parser.add_argument("-wf", "--wfirm", type=str, metavar="weight of FIRM score file", help="weight of FIRM score file", required=False)
    parser.add_argument("-wt", "--wit", type=str, metavar="weight of IT score file", help="weight of IT score file", required=False)
    parser.add_argument("-wp", "--wprob", type=str, metavar="weight of Prob score file", help="weight of Prob score file", required=False)
    parser.add_argument("-s", "--scoreCombine", type=str, metavar="output score file (combined results)", help="file name for output score (combined results)", required=False)
    parser.add_argument("-c", "--column", type=str, metavar="column number", help="column number in the table", required=False)
    parser.add_argument("-r", "--row_id", type=str, metavar="row_id number", help="row_id number in the db", required=False)
    parser.add_argument("-df", "--desc_file", type=str, metavar="feature description mapping file", help="feature description mapping file", required=False)
    parser.add_argument("-dsid", "--ds_id", type=str, metavar="source dataset id", help="source dataset id for training option", required=False)

    parser.add_argument("-u", "--uploadtype", type=str, metavar="upload type", help="data type", required=False)
    
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
    args = parser.parse_args()
    
    if args.number:
        num_to_show = eval(args.number)
    else:
        num_to_show  = 50
    if args.firm:
        FIRM_score_file = args.firm
    else:
        FIRM_score_file  = 'score_FIRM.txt'
    if args.it:
        IT_score_file = args.it
    else:
        IT_score_file  = 'score_IT.txt'
    if args.prob:
        Prob_score_file = args.prob
    else:
        Prob_score_file  = 'score_Prob.txt'
    if args.wfirm:
        w_FIRM = eval(args.wfirm)
    else:
        w_FIRM  = 1
    if args.wit:
        w_IT = eval(args.wit)
    else:
        w_IT  = 1
    if args.wprob:
        w_Prob = eval(args.wprob)
    else:
        w_Prob  = 1
    if args.scoreCombine:
        score_file_combine = args.scoreCombine
    else:
        score_file_combine  = 'score_combine.txt'
    if args.row_id:
        row_id_str = args.row_id
    else:
        row_id_str  = '0'
    if args.ds_id:
        ds_id = args.ds_id
    else:
        ds_id  = ''        
    if args.uploadtype:
        uploadtype = args.uploadtype
    else:
        uploadtype  = None
    if len(args.username)>0:
        username = args.username
    else:
        username  = None
    if len(args.password)>0:
        password = args.password
    else:
        password  = None 
    if args.desc_file:
        description_file=args.desc_file
    else:
        description_file='./tbd.txt'

    # mongo info for connection
    mongo_tuples=(args.ip_address, args.port, args.db_name, args.tb_name, username, password)
    
    return feat_importance_comb(row_id_str, ds_id, num_to_show, w_FIRM, w_IT, w_Prob
        , mongo_tuples, FIRM_score_file, IT_score_file, Prob_score_file, score_file_combine)

    
# ================================================================================== train () ============     
def feat_importance_comb(row_id_str, ds_id, num_to_show, w_FIRM, w_IT, w_Prob
        , mongo_tuples, FIRM_score_file, IT_score_file, Prob_score_file, score_file_combine): 
    human_verified=dict()
    all_verified=dict()

    print "INFO: ======= Combine all feature importance info ================" 
    
    # get feature importance voting data from db
    all_verified, human_verified=exec_sqlite.get_dict(row_id_str)
    #print "INFO: human_verified dict=",human_verified 
    #print "INFO: all_verified dict=",all_verified 

    ############begin##################
    with open(FIRM_score_file, 'r') as f:
        FIRM_score = f.readlines()
    with open(IT_score_file, 'r') as f:
        IT_score = f.readlines()
    with open(Prob_score_file, 'r') as f:
        Prob_score = f.readlines()
    
    # create file for one table here ======================
    dir_name=os.path.dirname(FIRM_score_file)
    coef_filename=os.path.join(dir_name,row_id_str+'_score_coef_comb.json')
    #print "INFO: combined fname=",coef_filename
    
    # get data from mongo
    key = "coef_arr"
    jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
    jstr_proj='{"value":1}'

    # get parent dataset's data
    if ds_id != row_id_str:
        jstr_filter='{"rid":'+ds_id+',"key":"'+key+'"}'

    doc=query_mongo.find_one_t(mongo_tuples, jstr_filter, jstr_proj)
    coef_arr = doc['value']
    #print "INFO: len(coef_arr)=",len(coef_arr)   
    
    # get sample count ===========
    key = "feat_sample_count_arr"
    jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
    jstr_proj='{"value":1}'
    
    # get parent dataset's data
    # need to chk if new feature
    if ds_id != row_id_str:
        jstr_filter='{"rid":'+ds_id+',"key":"'+key+'"}'
    
    doc=query_mongo.find_one_t(mongo_tuples, jstr_filter, jstr_proj)
    
    feat_sample_count_arr=None
    if not doc is None:
        feat_sample_count_arr = doc['value']
    
    
    combine_with_coef(row_id_str,coef_arr,FIRM_score,IT_score,Prob_score,coef_filename,feat_sample_count_arr)
    
    featurelist_FIRM = []
    featurelist_IT = []
    featurelist_Prob = []
    
    dic_all_columns = {}
    print "INFO: num to_show=",num_to_show
    for i in range(0, num_to_show):
        ####FIRM####
        str_in = FIRM_score[i]
        feature_id, score, descpt =str_in.split('\t', 3)
        featurelist_FIRM.append(feature_id)
        if not feature_id in dic_all_columns:
            dic_all_columns[feature_id]=descpt
        ####IT####
        str_in = IT_score[i]
        feature_id, score, descpt =str_in.split('\t', 3)
        featurelist_IT.append(feature_id)
        if not feature_id in dic_all_columns:
            dic_all_columns[feature_id]=descpt
        ####Prob####
        str_in = Prob_score[i]
        feature_id, score, descpt =str_in.split('\t', 3)
        featurelist_Prob.append(feature_id)
        if not feature_id in dic_all_columns:
            dic_all_columns[feature_id]=descpt
    
    list_i = [i+1 for i in range(0, num_to_show)]
    zipped_FIRM = zip(featurelist_FIRM, list_i)
    zipped_IT = zip(featurelist_IT, list_i)
    zipped_Prob = zip(featurelist_Prob, list_i)
    
    FIRM_dict = dict(zipped_FIRM)
    IT_dict = dict(zipped_IT)
    Prob_dict = dict(zipped_Prob)
    
    list_combine = featurelist_FIRM + featurelist_IT + featurelist_Prob
    list_unique = OrderedDict.fromkeys(list_combine).keys()
    #print list_unique, len(list_unique)
    
    #human_verified = {'188':7, '218':6} #####get human_verified from database, all click_number > 5 are human_verified###
    score_combine = {}
    for i in range (0, len(list_unique)):
        feat_id = list_unique[i]
        if feat_id in human_verified:
            print "INFO: found feat_id=",feat_id
            continue
        score = 0
        if feat_id in FIRM_dict:
            score = score + w_FIRM * FIRM_dict[feat_id]
        else:
            score = score + w_FIRM * (num_to_show + 1)
        if feat_id in IT_dict:
            score = score + w_IT * IT_dict[feat_id]
        else:
            score = score + w_IT * (num_to_show + 1)
        if feat_id in Prob_dict:
            score = score + w_Prob * Prob_dict[feat_id]
        else:
            score = score + w_Prob * (num_to_show + 1)
        score = score/float(3)
        #print feat_id, score
        
        #############add human feedback##########
        if feat_id in all_verified:
            click_number = all_verified[feat_id] #####get click number from database###
        else:
            click_number = 0
        #print feat_id, click_number, score
        #click_number = 3
        if click_number == 1:
            score = score - 2
        elif click_number == 2:
            score = score - 4
        elif click_number == 3:
            score = score - 10
        elif click_number == 4:
            score = score - 20
        elif click_number == 5:
            score = 0
        
        if score < 0:
            score = 0
        #print "***=",feat_id, click_number, score
        
        score_combine[feat_id] = score
    
    if os.path.exists(score_file_combine):
        try:
            os.remove(score_file_combine)
        except OSError, e:
            print ("Error: %s - %s." % (e.score_file_combine,e.strerror))
    
    
    combined_score=[]
    for x in sorted(score_combine.items(), key=operator.itemgetter(1)):
        (feat, score) = x
        #print "feat=",feat,",score=",score,",desc=",dic_all_columns[feat]
        
        description_str = dic_all_columns[feat]

        str01 = feat+"\t"+str(score)+"\t"+description_str
        combined_score.append(str01);
        with open(score_file_combine, "a") as f:
            f.write(str01)
    
    ### insert feature importance into mongoDB  ### 
    filter='{"rid":'+row_id_str+',"key":"feature_importance"}'
    upsert_flag=True
    jstr_insert = '{ "rid":'+row_id_str+',"key":"feature_importance", "value":{'
    
    # get parent dataset's data
    #if ds_id != row_id_str:
    #    jstr_insert='{"rid":'+ds_id+',"key":"feature_importance", "value":{'  
    #print "jstr_insert=",jstr_insert
    # write to option record. check if same feature?
    jstr_insert+='"ranking_measure":['
    for idx,line in enumerate(FIRM_score):         # FIRM
        if idx < num_to_show:
            jstr_insert+='"'+line.rstrip()+'",'
    jstr_insert= jstr_insert[:len(jstr_insert)-1]   # remove last ','
    jstr_insert+='],"probability":['
    for idx,line in enumerate(Prob_score):                     # Prob
        if idx < num_to_show:
            jstr_insert+='"'+line.rstrip()+'",'
    jstr_insert= jstr_insert[:len(jstr_insert)-1]   # remove last ','
    jstr_insert+='],"infomation_gain":['
    for idx,line in enumerate(IT_score):                       # IT
        if idx < num_to_show:
            jstr_insert+='"'+line.rstrip()+'",'
    jstr_insert= jstr_insert[:len(jstr_insert)-1]   # remove last ','
    jstr_insert+='],"combined":['
    for idx,line in enumerate(combined_score):                 # Combined
        if idx < num_to_show:
            jstr_insert+='"'+line.rstrip()+'",'
    jstr_insert= jstr_insert[:len(jstr_insert)-1]   # remove last ','
    jstr_insert+=']}}'
    jstr_insert=jstr_insert.replace("\t",",")

    #print "jstr_insert=",jstr_insert
    ## write to mongoDB.myml.dataset_info, ignore doc with duplicated key
    ret=query_mongo.upsert_doc_t(mongo_tuples,filter,jstr_insert,upsert_flag)
    print "INFO: Upsert count for feature importance=",ret    
    
    print 'INFO: Finished!'
    return 0

# combint to one table
def combine_with_coef(row_id_str,coef_arr,FIRM_list,IT_list,Prob_list,out_filename,feat_sample_count_arr):
    #print "INFO: combined fname=",out_filename
    if os.path.exists(out_filename):
        try:
            os.remove(out_filename)
        except OSError, e:
            print ("ERROR: %s - %s." % (e.out_filename,e.strerror))
    #print "INFO: len f=",len(FIRM_list),",len i=",len(IT_list),",len p=",len(Prob_list),",len c=",len(coef_arr)
    ig={}
    #for idx, val in enumerate(FIRM_list):
    for i in IT_list:
        vals=i[0:len(i)-1].split('\t')
        ig[vals[0]]=vals[1]
    pr={}
    for i in Prob_list:
        vals=i[0:len(i)-1].split('\t')
        pr[vals[0]]=vals[1]
    fi={}
    arr=[]
    if not feat_sample_count_arr is None and len(feat_sample_count_arr)>0:
        has_feat_sample_count=True
    else:
        has_feat_sample_count=False
    #for idx, val in enumerate(FIRM_list):
    for i in FIRM_list:
        # remove ending \n
        vals=i[0:len(i)-1].split('\t')
        # fid
        idx=vals[0]
        fi={}
        # coef, string, FIRM score, ig score, prob score
        #fi[idx]=[coef_arr[int(idx)-1],vals[2] ,vals[1], ig[idx],pr[idx]]
        fi["fid"]=idx
        fi["coef"]=coef_arr[int(idx)-1]
        if has_feat_sample_count:
            fi["feat_sample_count"]=feat_sample_count_arr[int(idx)-1]
        else:
            fi["feat_sample_count"]="n/a"
        fi["raw_str"]=vals[2]
        try:
            if np.isnan(float(vals[1])):
                fi["fmscore"]="NaN"
            else:
                fi["fmscore"]=float(vals[1])
        except:
            fi["fmscore"]=vals[1]
        try:
            if np.isnan(float(ig[idx])):
                fi["igscore"]="NaN"
            else:
                fi["igscore"]=float(ig[idx])
        except:
            fi["igscore"]=ig[idx]
        try:
            if np.isnan(float(pr[idx])):
                fi["pbscore"]="NaN"
            else:
                fi["pbscore"]=float(pr[idx])
        except:
            fi["pbscore"]=pr[idx]
        arr.append(fi)
    out={}
    out["data"]=arr
    #print "fi=",fi
    #print "out=",out
    # write output
    with open(out_filename, "w") as f:
        f.write(json.dumps(out))
        #print "INFO: file created." 

    f.close()
    return
    
if __name__ == '__main__':
    __description__ = "combine feature importance results"
    main()
