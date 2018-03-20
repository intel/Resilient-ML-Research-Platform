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
import zipfile, gzip

from argparse import ArgumentParser
from time import time

####pyspark#####
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import pydoop.hdfs as hdfs

####import our own library####
sys.path.append('./db')
import query_mongo
import ml_util
from ml_util import *
import zip_feature_util 

CONF_FILE='../../app.config' # at the base dir of the web
config=ConfigParser.ConfigParser()
config.read(CONF_FILE)

def main():
    
    parser = ArgumentParser(description=__description__)
    parser.add_argument("-shd", "--hdfs_src_dir", type=str, metavar="hdfs sourc folder", help="hdfs sourc folder", required=False)
    parser.add_argument("-shf", "--hdfs_src_fname", type=str, metavar="source hdfs filename", help="hdfs file name data source", required=False)
    parser.add_argument("-zfn", "--zfname", type=str, metavar="zip filename for hash list", help="zip filename for hash list", required=False)
    parser.add_argument("-ohd", "--hdfs_out_dir", type=str, metavar="output hdfs folder", help="output hdfs folder", required=False)
    parser.add_argument("-ohf", "--hdfs_out_fname", type=str, metavar="output hdfs filename", help="hdfs file name for data output", required=False)
    parser.add_argument("-old", "--out_dir", type=str, metavar="output local folder", help="output local folder", required=False
                , default ='.')
    parser.add_argument("-olf", "--out_filename", type=str, metavar="output local filename", help="output local filename", required=False
                , default ='retrieved')
    parser.add_argument("-r", "--row_id", type=str, metavar="row id", help="row_id number in the db", required=False)
    parser.add_argument("-srid", "--src_id", type=str, metavar="source dataset id", help="source dataset id to inherit data from", required=False)

    parser.add_argument('-hl','--hash_list', type=str, dest='hash_list', help='a list of (join field/hash , label)')
    parser.add_argument('-jfn','--join_field_name', type=str, dest='join_field_name', help='field name to join two dataframe'
                , default ='md5')
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
                , default =config.get('mongo', 'db_name'))
    parser.add_argument('-t','--tb_name', type=str, dest='tb_name', help='mongodb table name'
                , default =config.get('mongo', 'tb_name'))

    # auth
    parser.add_argument('-un','--username', type=str, dest='username', help='mongodb username'
                , default =config.get('mongo', 'username'))
    parser.add_argument('-pw','--password', type=str, dest='password', help='mongodb password'
                , default =config.get('mongo', 'password'))  
    # query 
    parser.add_argument("-jp", "--jstr_proj", type=str, metavar="projection json string for field selection (SELECT)" \
        , help="project json string for file selection (SELECT)", required=False)
    parser.add_argument("-jf", "--jstr_filter", type=str, metavar="filter json string (WHERE)" \
        , help="filter json string (WHERE); _id will be replaced by binary md5", required=False)
    # SQL for SqlSpark           
    parser.add_argument('-tbl','--tblname', type=str, dest='tblname', help='table name for sql script'
                , default ='srcTbl')
    parser.add_argument('-sql','--sql_script', type=str, dest='sql_script', help='sql script for data to parquet file'
                , default ='select * from srcTbl')
    
    args = parser.parse_args()
    
    if args.hdfs_src_dir:
        hdfs_src_dir = args.hdfs_src_dir
    else:
        hdfs_src_dir  = config.get('app', 'HADOOP_MASTER')+'/user/hadoop/yigai/sality_virut_zbot_backdoor_hash_000'
    if args.row_id:
        row_id_str = args.row_id
    else:
        row_id_str  = '88'
    if args.hdfs_out_dir:
        hdfs_out_dir = args.hdfs_out_dir
    else:
        hdfs_out_dir  = 'out_result'
    if args.src_id:
        src_id = args.src_id
    else:
        src_id  = ''
    if args.hdfs_src_fname:
        hdfs_src_fname=args.hdfs_src_fname
    else:
        hdfs_src_fname='out.parquet'
    if args.hdfs_out_fname:
        hdfs_out_fname=args.hdfs_out_fname
    else:
        hdfs_out_fname='out_inherited.parquet'
    if args.zfname:
        zfname=args.zfname
    else:
        zfname='~/mym/media/upload/test.zip'
        
    # hash_list is a list of tuple ('md5 string','label string')
    if args.hash_list:
        try:
            hash_list=eval(args.hash_list)
        except:
            e = sys.exc_info()[0]
            print "ERROR: ", e 
            return -1
    else:
        hash_list=[]
    if len(args.username)>0:
        username = args.username
    else:
        username  = None
    if len(args.password)>0:
        password = args.password
    else:
        password  = None
    if args.jstr_proj:
        jstr_proj = args.jstr_proj
    else:
        jstr_proj  = None
    if args.jstr_filter:
        jstr_filter = args.jstr_filter
    else:
        jstr_filter  = None

    # mongo info for connection
    mongo_tuples=(args.ip_address, args.port, args.db_name, args.tb_name, username, password)
    
    return extract_parquet( row_id_str, hdfs_src_dir, hdfs_src_fname, hdfs_out_dir, hdfs_out_fname
        , args.sp_master,config.get('spark', 'spark_rdd_compress'),config.get('spark', 'spark_driver_maxResultSize'), args.exe_memory, args.core_max
        , 'extract_dataset:'+row_id_str, args.join_field_name, zfname, args.tblname, args.sql_script
        , mongo_tuples
        , jstr_filter, jstr_proj, args.out_dir, args.out_filename)
        
# ================================================================================== extract_parquet ============ 
def extract_parquet( row_id_str, hdfs_src_dir, hdfs_src_fname,  hdfs_out_dir, hdfs_out_fname
    , sp_master, spark_rdd_compress, spark_driver_maxResultSize, sp_exe_memory, sp_core_max
    , jobname, join_field_name, zfname, tblname, sql_script
    , mongo_tuples
    , jstr_filter, jstr_proj, out_dir, out_filename):
    
    # extract dataframe from source parquet, save to output parquet and return unfounded list
    hash_lbl_list=save_inherited_ds(row_id_str, hdfs_src_dir, hdfs_src_fname, hdfs_out_dir, hdfs_out_fname
        , sp_master, spark_rdd_compress, spark_driver_maxResultSize, sp_exe_memory, sp_core_max
        , jobname, join_field_name, zfname , tblname, sql_script)
    
    print "INFO: unfounded list len=", len(hash_lbl_list)
    
    # support clean/dirty for now
    all_list_dict={}
    # find label from list: TBD 
    for i in hash_lbl_list:
        # label as key
        if i[1] not in all_list_dict:
            all_list_dict[i[1]]=[]
        # append to list
        all_list_dict[i[1]].append(i[0])
        
    print "INFO: all_list_dict len=", len(all_list_dict)
            
    # get_connection
    client, db, table=query_mongo.get_conn_t(mongo_tuples)

    # for mongo query
    if jstr_filter:
        json_filter=json.loads(jstr_filter)
    if jstr_proj:
        json_proj=json.loads(jstr_proj)

    
    # loop through all labels
    for label, h_list in all_list_dict.iteritems():
        print "INFO: lable=", label, ", count=", len(h_list)
        # prepare out file and out folder
        ofname=out_filename+"_"+label
        query_mongo.prepare_local_out_file(out_dir, ofname)
        total,has_data_count,no_data_count=query_mongo.query2gz(h_list, table, ofname, json_filter, json_proj, out_dir, label, label)
        if has_data_count == 0:
            print "WARNING: no data found for '"+label+"'"

    return 0

# ================================================================================== get hash list from zip file ============ 
# get hash list from zip file, return [(hash,label)]
def get_hash_list(hash_list_zfname):    
    file_list = []
    ret=[]
    # open hash list from the .zip file
    with  zipfile.ZipFile(hash_list_zfname, "r") as z:
        # get file list in .zip
        for filename in z.namelist():
            #print filename
            file_list.append(filename)
        print "INFO: file list=",file_list
        for file in file_list:
            # get label from folder in zip file
            name, label = file.split(".")
            label=label.strip().lower()
            #print "INFO: name.label=", name+"."+label
            
            # read zip file
            for line in z.read(file).split('\n'):
                line=line.strip().lower()
                if (len(line)>0):
                    #print (line,label)
                    ret.append((line,label))
    return ret
    
# ================================================================================== save_inherited_ds () ============ 
#  to retrieve parquet data from prior dataset and return new not found hash_list 
def save_inherited_ds(row_id_str, hdfs_src_dir, hdfs_src_fname, hdfs_out_dir, hdfs_out_fname
    , sp_master=config.get('spark', 'spark_master')
    , spark_rdd_compress=config.get('spark', 'spark_rdd_compress')
    , spark_driver_maxResultSize=config.get('spark', 'spark_driver_maxResultSize')
    , sp_exe_memory=config.get('spark', 'spark_executor_memory')
    , sp_core_max=config.get('spark', 'spark_cores_max')
    , jobname='extract_dataset:'
    , join_field_name='md5' , zfname=None, tblname='', sql_script='', hash_list=None
    ): 
    
    if len(zfname)>0:
        print "INFO: zip filename=", os.path.basename(zfname)
        hash_list=get_hash_list(zfname) 
 
    if len(hash_list)<=0:
        print "ERROR: hash list is required!"
        return -1
        
    if not row_id_str in jobname:
        jobname=jobname+row_id_str
        
    #hash_list=[ ('2','a'),('3','b'),('99','xx'),('101','yy') ]
    print 'INFO: hash_list len=',len(hash_list), ", type=", type(hash_list)
    
    # get_spark_context  
    sc=ml_util.ml_get_spark_context(sp_master
        , spark_rdd_compress
        , spark_driver_maxResultSize
        , sp_exe_memory
        , sp_core_max
        , jobname
        ) 
        
    t0 = time()
    sqlCtx = SQLContext(sc)
    # convert input list into a DF

    # TBD hardcode "_label_"
    sch=StructType([StructField(join_field_name,StringType(), True),StructField("_label_",StringType(), True)])    
    plist=sc.parallelize(hash_list)
    list_df=sqlCtx.createDataFrame(plist, sch)
    
    # debug only
    #list_df.show()
    
    # source filename
    src_fname=os.path.join(hdfs_src_dir,hdfs_src_fname)
    print "INFO: src fname for source dataset=", os.path.basename(src_fname),",join_field_name=",join_field_name
    
    sqlCtx = SQLContext(sc)
    src_df=sqlCtx.read.parquet(src_fname)
    #src_df.registerTempTable('pTbl')
         
    # get intersect DF
    src_df_rt=src_df.join(list_df,src_df[join_field_name]==list_df[join_field_name],'right_outer').cache()
    
    # existing data for reuse: 
    existing_df=src_df_rt.select(src_df['*']).where(src_df[join_field_name].isNotNull()).cache()
    
    count_df=existing_df.count()
    print "INFO: existing_df count=",count_df
    
    ''' debug only
    print "INFO: src_df="
    src_df.show()
    print "INFO: existing_df="
    existing_df.show()
    print "INFO: src_df_rt="
    src_df_rt.show()
    '''

    # generate out folder, clean up if needed
    try:
        hdfs.mkdir(hdfs_out_dir)
    except:
        e = sys.exc_info()[0]
        print "WARNING: ", e    
    # output hdfs file name
    out_fname=os.path.join(hdfs_out_dir,hdfs_out_fname)
    print "INFO: save data from inherited dataset to file=", os.path.basename(out_fname)
    # clean up existing hdfs file
    try:
        hdfs.rmr(out_fname)
    except:
        e = sys.exc_info()[0]
        print "WARNING: ", e  
    
    print "INFO: tbl=",tblname,",sql=",sql_script
        
    # convert to sql table
    existing_df.registerTempTable(tblname)
    df2= sqlCtx.sql(sql_script)
    # save as parquet
    df2.write.parquet(out_fname)
    
    #df2.printSchema()
    
    new_list=[]
    # new list which not in src DF for new retrival
    row_list=src_df_rt.select(list_df['*'],src_df[join_field_name]).where(src_df[join_field_name].isNull()).collect()
    new_list=[ (i[0],i[1])  for i in row_list]
    #print "row_list=",row_list
    #print "INFO: new_list len=",len(new_list)

    ''' debug only
    test4=sqlCtx.read.parquet(out_fname)
    print "reload parquet="
    test4.show()
    '''
    t1 = time()
    print 'INFO: running time: %f' %(t1-t0)
    print 'INFO: save_inherited_ds finished!'
    
    return new_list

    
if __name__ == '__main__':
    __description__ = "extract data from existing dataframe and query new ones"
    main()
