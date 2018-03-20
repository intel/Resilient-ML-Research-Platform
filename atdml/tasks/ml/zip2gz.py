#! /usr/bin/python
'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''
# standard library imports
from argparse import ArgumentParser
import os
import re
import sys, ConfigParser
import zipfile, gzip
import collections,datetime
from time import time
import ujson, json
from pydoop import hdfs

CONF_FILE='../../app.config' # at the base dir of the web
config=ConfigParser.ConfigParser()
config.read(CONF_FILE)
HDFS_RETR_DIR = config.get("env","HDFS_RETR_DIR")


def arg_parser(parser):
    parser.add_argument("-odfs", "--dfs_folder", type=str, metavar="hdfs folder for output", help="hdfs folder for output", required=False)

    parser.add_argument("-zif", "--in_zipfname", type=str, metavar="input zip file names", help="zip file name of samples", required=False)
    parser.add_argument("-zod", "--outdir", type=str, metavar="output dir", help="the dir for output files", required=False)
    parser.add_argument("-zfn", "--outfname", type=str, metavar="output dir", help="the dir for output files", required=False)
    
    #dataset info
    parser.add_argument("-r", "--row_id", type=str, metavar="row id", help="row_id number in the db", required=False)

    return parser.parse_args()

# convert .zip file to Spark friendly .gz file
# format for .zip file:
#  first level directory name as the label name
#    allow multiple files in each folder
#    each file contains raw string/log from trace 
#  transform each file into one row of string in output .gz file 
#    for each file in input .zip file: replace "\t" with " ", "\r\n" and "\n" with "\t" and then add "\n" at the end
#    prefix each row with label, filename(md5), date of file; separated by "\t"
#  for every 100 sample files, create a new .gz file
def main():  # ============= =============  ============= =============
    # parse arguments
    parser = ArgumentParser(description=__description__)
    args = arg_parser(parser)

    if args.in_zipfname:
        in_zipfname = args.in_zipfname
    else:
        in_zipfname  = 'data_test.zip'
    if args.outdir:
        outdir = args.outdir
    else:
        outdir = 'out'
    if args.outfname:
        outfname = args.outfname
    else:
        outfname = 'outfname'
    if args.dfs_folder:
        dfs_folder = args.dfs_folder
    else:
        dfs_folder = None

    if args.row_id:
        row_id_str = args.row_id
    else:
        row_id_str  = "0"

    # log time ================================================================ ================
    t0 = time()

    print "outdir=",outdir
    
    # get input zip file name
    outfname = os.path.basename(in_zipfname)
    # get file name and its extension suffix
    (root, ext) = os.path.splitext(outfname)
    
    # set output file name
    outfname=root+".gz"
    # get output file handle
    zout=create_zfile(outdir, outfname)
    
    # input file is .zip file with folders inside
    zin = zipfile.ZipFile(in_zipfname, "r")
    count=0
    
    folder_list = []
    
    # open each file in .zip file, parse it  and save to .gz file
    #for filename in zin.namelist():
    for info in zin.infolist():
        # get filename from zin
        filename=info.filename
        #dd=info.date_time
        #print "f=",filename,", dt=",datetime.datetime(*dd)

        meta_list=[]
        # get folder name
        folder,name = filename.split('/')
        #print "--",folder," --",filename
        
        # assume first level folders are labels
        # collect labels
        if not folder in folder_list:
            folder_list.append(folder)
            
        # transform here ======================
        if len(name)>0:     # exclude folder name
            content = zin.read(filename)
            if len(content)<=0:
                print "Content not found for ["+filename+"]"
            else:
                # count files
                count=count+1
                # label
                meta_list.append(folder)
                # md5;  assume file name is md5
                bname=os.path.basename(filename)
                (namep,ext)=os.path.splitext(bname)
                meta_list.append(namep)
                # date of file
                meta_list.append(str(datetime.datetime(*info.date_time)))
                #print "meta_list=",meta_list
                #print "content=",len(content)," type=",type(content)
                #print "bname=",bname
                
                # for zip file; write to different files
                #zout.writestr(bname, format_content(meta_list, content))
                
                # write to .gz file; 
                zout.write(format_content(meta_list, content))
                
                # allow 100 samples in .gz file; create the other file
                if count%100==0:
                    zout.close()
                    outfname=root+"_"+str(count)+".gz"
                    zout=create_zfile(outdir, outfname)
                    
    zout.close()
    zin.close()
    #print "folder_list=",folder_list
    
    #upload to HDFS
    if dfs_folder:
        # clean up folder
        dfs_folder=os.path.join(HDFS_RETR_DIR, dfs_folder)
        print "dfs_folder=",dfs_folder
        try:
            hdfs.rmr(dfs_folder)
        except:
            e = sys.exc_info()[0]
            print "Warning: delete hdfs error: ",e
            pass
            
        try:
            hdfs.put(outdir, dfs_folder)
        except:
            e = sys.exc_info()[0]
            print "Error: Put files error." ,e
        
    t1 = time()
    print 'running time: %f' %(t1-t0)
    return 0
    
    '''
python ./ml/zip2gz.py -zif ../../media/upload/atd_rdata_small.zip -zod /home/django/myml/media/tmpdata/atd01 -odfs atd01
hdfs dfs -rm /user/hadoop/upload/data_retrieved/atd01/*
hdfs.put("./atd01","/user/hadoop/upload/data_retrieved")
hdfs.rmr("/user/hadoop/upload/data_retrieved/atd01")
    '''

    
    
# open .gz file for write  ============================== ==============================
def create_zfile(dir, fname):
    # clean up folder?    
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except:
        e = sys.exc_info()[0]
        print "Warning: makedirs failed: ",e
        pass
        
    ffname=os.path.join(dir,fname)
    print "zip ffname=",ffname
    
    # clean up file
    try:
        if os.path.isfile(ffname):
            os.remove(ffname)
            print "file ["+ffname+"] removed."
    except:
        e = sys.exc_info()[0]
        print "Warning: delete file error: ",e
        pass
        
    # for zip
    #zout = zipfile.ZipFile(ffname, "w",compression=zipfile.ZIP_DEFLATED)  
    
    # create .gz file
    zout = gzip.open(ffname, "wb") 
    
    return zout
 

 # add meta data and put every thing into one line (for Spark) ==============================
def format_content(meta_list, content):
    if content is None or len(content)<=0:
        return ""

    m=""
    for i in meta_list:
        m=m+i+"\t"
    content = content.replace("\t"," ")
    content = content.replace("\r\n","\t")
    content = content.replace("\n","\t")
    return m+content+"\n"
    #x = content.split('\t')
    #return m+x[0]+"\t=="+x[1]
    

    

 


    
    
    


if __name__ == '__main__':
    __description__ = "text file preprocessing and feature extraction"
    main()
