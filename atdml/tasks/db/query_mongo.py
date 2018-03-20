'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''
import httplib, urllib, base64
import time
import pprint
import requests,sys,getopt,  ConfigParser
import os, time, shutil, binascii, datetime, chardet
import zipfile, gzip
import pymongo
import isodate
from argparse import ArgumentParser

import pymongo
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import bson, json
from pprint import pprint as pp
import datetime

from bson import json_util
from bson.json_util import dumps
from bson.json_util import loads
import ast

CONF_FILE='/home/django/myml/app.config' # at the base dir of the web
config=ConfigParser.ConfigParser()
config.read(CONF_FILE)

ITEM_PER_GZFILE=int(config.get('mongo','ITEM_PER_GZFILE'))

    
# http://api.mongodb.org/python/current/api/pymongo/collection.html
def main():
    
    # Input output setup
    parser = ArgumentParser(description=__description__)
    parser.add_argument("-o", "--output", type=str, metavar="output dir", help="the output dir", required=False)
    parser.add_argument("-z", "--filename", type=str, metavar="filename", help="filename of hash list in zip or plain text", required=False)
    # 
    parser.add_argument("-m", "--mode", type=str, metavar="input mode", help="", required=False)
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
    # insert docs
    parser.add_argument("-ji", "--jstr_insert", type=str, metavar="json string for insert" \
        , help="son string for insert; ", required=False)
    parser.add_argument("-sh", "--str_hash", type=str, metavar="hash string for query" \
        , help="get a doc by hash string", required=False)

        
    args = parser.parse_args()
    
    if args.output:
        out_dir = args.output
    else:
        out_dir = None

    if len(args.username)>0:
        username = args.username
    else:
        username  = None
    if len(args.password)>0:
        password = args.password
    else:
        password  = None
    if args.filename:
        ifilename = args.filename
    else:
        ifilename  = 'hash_pdf_min.zip'
    if args.jstr_proj:
        jstr_proj = args.jstr_proj
    else:
        jstr_proj  = None
    if args.jstr_filter:
        jstr_filter = args.jstr_filter
    else:
        jstr_filter  = None
    if args.jstr_insert:
        jstr_insert = args.jstr_insert
    else:
        jstr_insert  = None
    if args.str_hash:
        str_hash = args.str_hash
    else:
        jstr_insert  = None

    if args.mode:
        mode = args.mode
    else:
        mode  = "0"  # default for hash list
    ################################begins################################
    
    # clean up output folder
    if out_dir:
        if os.path.exists(out_dir):
            try:
                shutil.rmtree(out_dir)
            except:
                print 'WARNING: Drop dir tree error'
        # create output folder
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    
    ret=-1
    #print mode
    # dispatch by mode    
    if mode=="hash_ziplist": # get docs by hash list (_id) in a zip file; designed for web 
        ret=query_by_hash_ziplist(args.ip_address, args.port, args.db_name,args.tb_name, username,password
            , ifilename, jstr_filter, jstr_proj, out_dir)
    if mode=="hash_list": # get docs by hash list (_id); designed for web 
        ret=query_by_hash_list_file(args.ip_address, args.port, args.db_name,args.tb_name, username,password
            , ifilename, jstr_filter, jstr_proj, out_dir)
    elif mode == "hash": # get a doc by hash (_id); designed for web 
        ret=query_by_hash(args.ip_address, args.port, args.db_name,args.tb_name, username,password
            ,str_hash, jstr_proj, out_dir)
    elif mode == "insert_ik": # insert record & ignore key ; designed for web upload feature mapping
        ret=insert_docs(args.ip_address, args.port, args.db_name,args.tb_name, username,password
            ,jstr_insert)
    elif mode == "insert": # insert doc
        ret=insert_many(args.ip_address, args.port, args.db_name,args.tb_name, username,password
            ,jstr_insert)
    elif mode == "insert_gz": # insert docs from .gz files
        ret=insert_from_gz(args.ip_address, args.port, args.db_name,args.tb_name, username,password
            ,ifilename)
    elif mode == "query": # query and get cursor, out_dir is the key to output to file(s)
        return select_cursor(args.ip_address, args.port, args.db_name,args.tb_name, username,password
            ,jstr_filter,jstr_proj, out_dir, None, None)    
    elif mode == "query_outzip": # query and output as zip file
        #print "in query_outzip"
        return select_cursor(args.ip_address, args.port, args.db_name,args.tb_name, username,password
            ,jstr_filter,jstr_proj, out_dir, ifilename, "gz")    
    #client.close()
    return ret
##################################################################################

def get_conn_t(mongo_tuples):
        return get_conn(mongo_tuples[0], mongo_tuples[1], mongo_tuples[2] 
        , mongo_tuples[3], mongo_tuples[4], mongo_tuples[5])

def get_conn(ip, port, db,tb_name, username,password):
    #print "ip=",ip,",port=",port,",db=",db,",tb=",tb_name,"usr=",username
    # connect to MongoDB ======================================
    client = MongoClient(ip, int(port))
    if username and password:
        #print "Auth into DB"
        client[db].authenticate(username,password)
    db = client[db]
    table = db[tb_name]
    return client, db, table

# create folder if not exist, delete file if exist
def prepare_local_out_file(out_dir, out_fname):
    # create output folder if not exist
    # remove file if exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if out_fname and len(out_fname)>0:
        if os.path.isfile(out_fname):
            try:
                os.remove(out_fname)
            except:
                print 'WARNING: Remove file error!'


    
##################################################################################
# upsert_doc() ignore duplicates            
def upsert_doc(ip, port, db,tb_name, username,password,jstr_filter,jstr_replace,upsert_flag): 
    client, db, table=get_conn(ip, port, db,tb_name, username,password)
    json_insert=json.loads(jstr_replace)
    filter=json.loads(jstr_filter)
    rc=0
    #print doc
    try:
        rt=table.replace_one(filter,json_insert,upsert_flag)
        rc+=1
    except Exception as e:
        print "ERROR: in upsert_doc(): " , e
    client.close()
    return rc

##################################################################################
# upsert_doc()             
def upsert_doc_t(mongo_tuples,jstr_filter,jstr_replace,upsert_flag): 
    return upsert_doc(mongo_tuples[0], mongo_tuples[1], mongo_tuples[2] 
        , mongo_tuples[3], mongo_tuples[4], mongo_tuples[5]
        ,jstr_filter,jstr_replace,upsert_flag)
    
##################################################################################
# insert_one() ignore duplicates            
def insert_doc(ip, port, db,tb_name, username,password,jstr_insert,table): 
    client=None
    if not table:
        client, db, table=get_conn(ip, port, db,tb_name, username,password)
    
    json_insert=json.loads(jstr_insert)
    #hash_lower=json_insert['FileInfo']['md5'].lower()
    #json_insert["_id"]=bson.binary.Binary(binascii.a2b_hex(hash_lower))           
    
    rc=0
    #print doc
    try:
        rt=table.insert_one(json_insert)
        rc+=1
    except Exception as e:
        #pass
        print "ERROR: in insert_doc()", e
    if client:
        client.close()
    return rc


# for loop to call insert_one(), ignore duplicates            
def insert_docs(ip, port, db,tb_name, username,password,jstr_insert): 
    client, db, table=get_conn(ip, port, db,tb_name, username,password)
    json_insert=json.loads(jstr_insert)
    rc=0
    for doc in json_insert:
        #print doc
        try:
            rt=table.insert_one(doc)
            rc+=1
        except DuplicateKeyError as e:
            pass
            #print "Warning: key duplicated" #% e
    client.close()
    return rc
##################################################################################
    
# for web to upload feature mapping file
def insert_many(ip, port, db,tb_name, username,password,jstr_insert): 
    client, db, table=get_conn(ip, port, db,tb_name, username,password)
    json_insert=json.loads(jstr_insert)
    try:
        ret=table.insert_many(json_insert)    
    except DuplicateKeyError as e:
        print "WARNING: key duplicated" #% e
    client.close()
    return ret
##################################################################################

# for web to delete
def delete_many(mongo_tuples, table, jstr_filter): 
    close_conn="N"
    if not table and mongo_tuples and len(mongo_tuples)>0 :
        client, db, table=get_conn_t(mongo_tuples)
        close_conn="Y"
    try:
        json_filter=json.loads(jstr_filter)
        ret=table.delete_many(json_filter)    
    except Error as e:
        print "WARNING: error in delete_many" #% e
    # clean up
    if close_conn=="Y":
        client.close()   
    return ret
##################################################################################


# query one doc by hash/_id & projection
def query_by_hash_t(mongo_tuples, str_hash, jstr_proj, out_dir, table=None):
    return query_by_hash( mongo_tuples[0], mongo_tuples[1], mongo_tuples[2] 
        , mongo_tuples[3], mongo_tuples[4], mongo_tuples[5]
        , str_hash, jstr_proj, out_dir, table )

# query one doc by hash/_id & projection
def query_by_hash(ip, port, db, tb_name, username, password, str_hash, jstr_proj, out_dir, table=None):

    if table is None:
        client, db_obj, table=get_conn(ip, port, db,tb_name, username,password)
    json_filter={}
    json_proj={}
    # clean up md5 data
    if len(str_hash)>32:
        str_hash =str_hash[len(str_hash)-32:]    
    if jstr_proj:
        json_proj=json.loads(jstr_proj)
    hash_lower=str_hash.lower()    
    json_filter["_id"]=bson.binary.Binary(binascii.a2b_hex(hash_lower))

    doc=select_onedoc_in(table, json_filter, json_proj)
    #print doc
    
    if out_dir:
        ###### dump to string
        jj = dumps(doc)
        #outfile = open(os.path.join(out_dir,hash_lower+".json"), "w")
        # size is big; need to zip it
        outfile = gzip.open(os.path.join(out_dir,hash_lower+".gz") ,'ab')
        outfile.write(jj)
        outfile.close()
    return doc
    #python query_mongo.py -o /tmp/562retrieve -m hash -sh 0xd429210c3c7bd201ccdcce3bcca65acb
    #-jp '{"FileInfo.md5":1,"FileInfo.DateAdded":1,"FileInfo.filetype":1,"METADATA.data.features":1,"METADATA.data.pegeometry":1,"_id":1}' 
##################################################################################
    
# query by filter & projection
def select_cursor(ip, port, db,tb_name, username,password, jstr_filter, jstr_proj, out_dir, filename, mode):  
    # connect to db
    client, db, table=get_conn(ip, port, db,tb_name, username,password)
    # conver string to json
    json_filter={}
    json_proj={}

    if jstr_filter:
        json_filter=json.loads(jstr_filter)
    if jstr_proj:
        json_proj=json.loads(jstr_proj)
    #print json_filter
    # flush stdout
    sys.stdout.flush()     
    #get cursor
    cursor=table.find(json_filter, json_proj) #.limit(2)
    
    # dump to file
    if out_dir:
        #print "out here"
        if filename==None:
            filename="out"
        ret=dump2files(cursor, out_dir, filename, mode,ITEM_PER_GZFILE)
        cursor.close()
        return ret
        
    return cursor
##################################################################################

# output as .gz or .json files per "item_per_file"" doc
def dump2files(cursor, out_dir, filename, mode, item_per_file):
    # get filename remove extension
    if filename.endswith('.gz'):
        filename=filename[:len(filename)-3]
    elif filename.endswith('.json'):
        filename=filename[:len(filename)-5]
    
    # open a .gz file for write/append 
    totalc=0
    # output gz file
    file =None
    fname =None
    ctotal=cursor.count()
    for doc in cursor:
        # open a new file per "item_per_file" doc
        if item_per_file>0 and totalc % item_per_file ==0:
            if file:
                file.close()
            if mode == "gz":
                fname=out_dir+os.sep+ filename+"_"+str(totalc)+".gz"
                # open .gz file for append/binary
                file=gzip.open(fname,'ab')
            else:
                fname=out_dir+os.sep+ filename+"_"+str(totalc)+".json"
                file=open(fname,'ab')
            print ''
            print "INFO: Output fname=",fname
        elif item_per_file <= 0:
            if mode == "gz":
                fname=out_dir+os.sep+ filename+".gz"
                file=gzip.open(fname,'ab')
            else:
                fname=out_dir+os.sep+ filename+".json"
                file=open(fname,'ab')
                
        # TBD: how to input the label
        doc["label"]="unknown"
        
        if totalc % 100 == 0:
            print '' # echo progress
            print "INFO: ",totalc, "/", ctotal, datetime.datetime.now()
        elif totalc % 10 == 0:
            print '.', # echo progress
        sys.stdout.flush() 
                
        ###### dump to string
        jj = dumps(doc)
        totalc +=1
        # spark didn't like .zip file
        file.write(jj)
        file.write('\n')   

    file.close()   
    print "INFO: Total count=", totalc  
    return totalc
    
##################################################################################
# query one doc by filter & projection
def find_one_t(mongo_tuples, jstr_filter, jstr_proj):
    return find_one(mongo_tuples[0], mongo_tuples[1], mongo_tuples[2] 
        , mongo_tuples[3], mongo_tuples[4], mongo_tuples[5] 
        , jstr_filter, jstr_proj)

def find_one(ip, port, db, tb_name, username, password, jstr_filter, jstr_proj):
    client, db, table=get_conn(ip, port, db,tb_name, username,password)
    if jstr_filter:
        json_filter=json.loads(jstr_filter)
    if jstr_proj:
        json_proj=json.loads(jstr_proj)
    return table.find_one(json_filter, json_proj)

    
##################################################################################
# internal for getting one doc
def select_onedoc_in(table, json_filter, json_proj):   
    doc=table.find_one(json_filter, json_proj)
    return doc

##################################################################################
# query by a hash list in .txt file and save by query2gz()
def query_by_hash_list_file(ip, port, db, tb_name, username, password, filename, jstr_filter, jstr_proj, out_dir):
    client, db, table=get_conn(ip, port, db,tb_name, username,password)
    json_filter={}
    json_proj={}

    if jstr_filter:
        json_filter=json.loads(jstr_filter)
    if jstr_proj:
        json_proj=json.loads(jstr_proj)
    file = open(filename,"r")
    
    # set filename as label 
    n,label = os.path.basename(filename).split(".")
    print "INFO: ",filename, n,label

    lines = file.readlines()
    print lines
    #label="" #TBD
    query2gz(lines, table, filename, json_filter, json_proj, out_dir,label,"")
    client.close()


##################################################################################
# retrieve doc and save to .gz file
def query2gz(hash_list, table, out_filename, json_filter, json_proj, out_dir, label, echo_prefix):
    gzipoutfile=None
    gzipoutfilename=None
    total = 0   ###total number of samples for processing ###
    has_data_count = 0   ###number of samples has data###
    no_data_count = 0   ####number of samples has NO data####
    hash_count=len(hash_list)
    flag_print_fname=0
    
    for hash in hash_list:
        hash=hash.strip()
        
        if len(hash)>1:         ####if not an empty line
            # clean up md5 data
            if len(hash)>32:
                hash =hash[len(hash)-32:]
                #print '>32'
            
            # open .gz file for output
            if has_data_count % ITEM_PER_GZFILE ==0 :
                if gzipoutfile and os.path.isfile(gzipoutfilename):
                    gzipoutfile.close()

                #os.path.join() didn't work with "e:\"
                gzipoutfilename=os.path.join(out_dir, out_filename+"_"+str(has_data_count)+".gz")
                if flag_print_fname==0: # avoid multiple print
                    print ''
                    print "INFO: Output filename=",gzipoutfilename
                    flag_print_fname=1
                gzipoutfile=gzip.open(gzipoutfilename,'ab')
                
            hash_lower = hash.lower()
            # incr count
            total = total + 1

            # query to get record 
            doc=None

            json_filter["_id"]=bson.binary.Binary(binascii.a2b_hex(hash_lower))   

            try: 
                doc=select_onedoc_in(table, json_filter, json_proj)
            except Exception as e: 
                print "retry",
                try: # retry once
                    doc=select_onedoc_in(table, json_filter, json_proj)
                except Exception as ei: 
                    print "ERROR:", ei

            if doc == None:
                print "x",  # echo progress
                print hash_lower
                no_data_count = no_data_count +1
                continue
            else: # reset flag for fname printing
                flag_print_fname=0
            if has_data_count % 100 == 0:
                print '' # echo progress
                print 'INFO: --- '+echo_prefix+"-"+str(total)+'/'+str(hash_count)+' --- ',
                print datetime.datetime.now()
            else:
                print '.', # echo progress
            sys.stdout.flush()    
            has_data_count = has_data_count + 1
            
            #add label into data TBD?
            doc['_label_']=label

            ###### dump to string
            jj = dumps(doc)

            # spark didn't like .zip file
            if gzipoutfile:
                gzipoutfile.write(jj)
                gzipoutfile.write('\n')

            #print "Sample number: ", total
        # end if len > 0
    # close .gz file
    if gzipoutfile:
        gzipoutfile.close() 
        
    # check file size. if zero then remove it
    if has_data_count == 0:
        print "INFO: ",gzipoutfilename,"has no record."
        os.remove(gzipoutfilename)
        
    print ""
    print "INFO: Total count for list [",label,"]: ", total
    print "INFO: Not found count ", no_data_count
    print "INFO: Fount count: ", has_data_count        
    return total,has_data_count,no_data_count
    
##################################################################################
# query by a hash list in .zip file and save by query2gz()
def query_by_hash_ziplist(ip, port, db, tb_name, username, password, ifilename, jstr_filter, jstr_proj, out_dir):        
    client, db, table=get_conn(ip, port, db,tb_name, username,password)
    json_filter={}
    json_proj={}

    if jstr_filter:
        json_filter=json.loads(jstr_filter)
    if jstr_proj:
        json_proj=json.loads(jstr_proj)
        
    # open hash list from the .zip file
    z = zipfile.ZipFile(ifilename, "r")
    # get file list in zip file
    file_list = []
    
    # output gz file
    gzoutfile =None
    gzoutfilename =None
    
    print "INFO: ======= Starting download ==============="
    # get file list in .zip
    for filename in z.namelist():
        #print filename
        file_list.append(filename)
    file_count=len(file_list)
    curr_file_count=0
    print "INFO: Total folder count in zip file=",file_count
    ret=0

    # get data for one list at a time
    for file in file_list:
        curr_file_count=curr_file_count+1
        print "INFO: -- ",str(curr_file_count),"/",str(file_count),". Filename: ", file
        
        # get label from folder in zip file
        name, label = file.split(".")
        #print name, label
        
        # read zip file
        content = z.read(file)
        
        # get hash list
        hash_list = [hashes.strip() for hashes in get_line(content)]
        hash_count=len(hash_list)
        #print hash_list
        print "INFO: hash_count=",hash_count
        
        total,with_log_number,not_found_number=query2gz(hash_list, table, file, json_filter, json_proj, out_dir, label
            ,str(curr_file_count)+"/"+str(file_count))
        if total==0:
            ret=-1

    # end for loop
    client.close()
    return ret
##################################################################################

##################################################################################
# TBD insert data from .gz file
def insert_from_gz(ip, port, db, tb_name, username, password, dirname): 
    file_list = [ os.path.join(dirname,f) for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname,f))]
    print "INFO: file_list=",file_list

    client, db, table=get_conn(ip, port, db,tb_name, username,password)
    
    for file in file_list:
        # open hash list from the .zip file
        gzfile = gzip.GzipFile(file, "r")
        lines=[line for line in gzfile.read().split('\n') if len(line)>0]
        for line in lines:
            print ">>",line

##################################################################################
def get_line(s):
    x = s.split('\n')
    for xx in x:
        if len(xx) > 1:
            yield xx

##################################################################################
if __name__ == '__main__':
    __description__ = "Access MongoDB data"
    main()
    print "INFO: END"