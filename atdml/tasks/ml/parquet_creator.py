#! /usr/bin/python
'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''


# standard library imports
import argparse
import calendar
import itertools
import md5
import os
import re
import sys, ConfigParser
import time, socket

# third party library imports
from pyspark import SparkContext
from pyspark import StorageLevel
from pyspark.sql import SQLContext
import ujson
import pydoop.hdfs as hdfs


CONF_FILE='../../app.config' # at the base dir of the web
config=ConfigParser.ConfigParser()
config.read(CONF_FILE)

def read_args(args):

    usage_str = '"This script combines .gz archives in a folder into a one parquet file by Spark jobs"'
    parser = argparse.ArgumentParser(usage_str)
    parser.add_argument("-r", "--row_id", type=str, metavar="row id", help="row_id number in the db", required=False)
    # input/output param
    parser.add_argument('-hd', '--hd_master', type=str, dest='hd_master', help='hadoop master url'
                , default = config.get('app', 'HADOOP_MASTER'))
    parser.add_argument('-d', '--src_dir', type=str, dest='src_dir', help='source dir'
                , default = '/user/hadoop/cyang8/data_pdf_min')
    parser.add_argument('-fe','--src_files', type=str, dest='src_files', help='source data filename with wildcard'
                , default ='*.gz')
    parser.add_argument('-o','--out_dir', type=str, dest='out_dir', help='path to output parquet file'
                , default ='out.parquet')
    # Spark params            
    parser.add_argument('-sp','--sp_master', type=str, dest='sp_master', help='spark.master'
                , default =config.get('spark', 'SPARK_MASTER'))
    parser.add_argument('-em','--exe_memory', type=str, dest='exe_memory', help='spark.executor.memory'
                , default =config.get('spark', 'spark_executor_memory'))
    parser.add_argument('-cm','--core_max', type=str, dest='core_max', help='spark.cores.max'
                , default =config.get('spark', 'spark_cores_max'))
    # SQL for SqlSpark           
    parser.add_argument('-tbl','--tblname', type=str, dest='tblname', help='table name for sql script'
                , default ='srcTbl')
    parser.add_argument('-sql','--sql_script', type=str, dest='sql_script', help='sql script for data to parquet file'
                , default ='select * from srcTbl')
    return parser.parse_args(args)




def main():
    # parse arguments
    print "INFO: creating parquet ..."
    args = read_args(sys.argv[1:])
    
    SparkContext.setSystemProperty('spark.rdd.compress', config.get('spark', 'spark_rdd_compress'))
    SparkContext.setSystemProperty('spark.driver.maxResultSize', config.get('spark', 'spark_driver_maxResultSize'))
    SparkContext.setSystemProperty('spark.executor.memory', args.exe_memory)
    SparkContext.setSystemProperty('spark.cores.max', args.core_max)
    sc = SparkContext(args.sp_master, 'parquet_creator:'+str(args.row_id))
    sqlCtx = SQLContext(sc)

    # load json obj from file to srdd
    ifname=args.hd_master+os.path.join(args.src_dir, args.src_files)
    
    df=sqlCtx.read.json(ifname)

    out_fname=args.hd_master+os.path.join(args.src_dir, args.out_dir)
    print "INFO: out_dir="+args.out_dir
    # clean up existing hdfs file
    try:
        hdfs.rmr(out_fname)
    except:
        e = sys.exc_info()[0]
        print "WARNING: ", e  
    
    # convert dataframe format
    print "INFO: SQL=",args.sql_script
    df.registerTempTable(args.tblname)
    df2= sqlCtx.sql(args.sql_script)
    # save as parquet
    df2.write.parquet(out_fname)
    
    df2.printSchema()

if __name__ == '__main__':
    main()
