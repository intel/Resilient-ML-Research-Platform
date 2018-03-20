#! /usr/bin/python
'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''

from argparse import ArgumentParser
import sys, ConfigParser
#####import for sqlite ####
sys.path.append('./db')
import exec_sqlite
import sqlite3,datetime
import os, shutil, glob, re

def main():
    parser = ArgumentParser(description=__description__)
    parser.add_argument("-r", "--row_id", type=str, help="row_id number in the db"
        , dest='rid', default ="0", required=False)
    parser.add_argument('-md', '--mday', type = str, help = 'filter file by modified days'
        , dest='mday', default ="14", required =False)
    parser.add_argument('-pttn', '--tgt_patterns', type = str, help = 'target filename pattern'
        , dest='tgt_patterns', default ="log/*.log,tmpdata/*retrieve,result/*", required =False)
    parser.add_argument('-dir', '--tgt_dir', type = str, help = 'directory of target files'
        , dest='tgt_dir', default ="../../media", required =False)
    parser.add_argument('-db', '--db_file', type = str, help = 'directory of target files'
        , dest='db_file', default ="../../db.sqlite3", required =False)
    parser.add_argument('-act', '--action', type = str, help = 'action applied to target files'
        , dest='action', default ="del_by_day", required =False)
    parser.add_argument('-dg', '--debug', type = str, help = 'print file list only'
        , dest='debug', default ="Y", required =False)
    args = parser.parse_args()

    if args.action:
        print "in action=" ,args.action,",dir=",args.tgt_dir,",pattern=",args.tgt_patterns,",mday=" \
            ,args.mday,",debug=",args.debug
        if args.action == "del_by_day": #  python ./ml/file_mgr.py -act del_by_day -md 14 -pttn "log/*predict.log,tmpdata/*retrieve" -dg Y 
            del_by_day(args.tgt_dir,args.tgt_patterns,args.mday,args.db_file,args.debug)
        elif args.action == "del_by_id":  #  python ./ml/file_mgr.py -r 945 -act del_by_id -dg Y
            del_by_id(args.tgt_dir,args.tgt_patterns,args.rid,args.db_file,args.debug)
        elif args.action == "del_by_db_ex":#  python ./ml/file_mgr.py -act del_by_db_ex -dg Y
            del_by_db_existance(args.tgt_dir,args.tgt_patterns,args.db_file,args.debug)
            #query_db()

# clean up files without id in db
def del_by_db_existance(tgt_dir,tgt_patterns, db_file,debug):            
    # for each file pattern, get id and find record
    for p in tgt_patterns.split(','):
        # delete file log/<id>predict.log
        filename=os.path.join(tgt_dir,p)
        #print "ls ",filename
        # get a list of filenames
        list=glob.glob(filename)
        # for each file get id and query db
        for f in list:
            id=None
            qret=None
            fname=os.path.basename(f)
            try:
                # get id from filename
                id=re.search(r'\d+',fname).group()
                # query db
                sql="select id from atdml_document where id="+id+";"
                # check if record exists
                qret=exec_sqlite.query_db(sql)
                # if not record found, drop the file
                if not qret is None and len(qret)==0:
                    print "id=",id #,"f=",fname
                    #if debug=="N":
                    delete_a_file_or_dir(f, debug)
                    #else:
                     #   print "rm",f
                    #break
                #print "q=", qret
            except:
                pass
                continue
            
# clean up invalid prediction file (after re-train)   ==================================== 
def del_by_id(tgt_dir,tgt_patterns, rid, db_file,debug):             

    sql=" select p.id,p.file_type,p.processed_date from atdml_document p join atdml_document d on p.train_id=d.id \
    where p.train_id >0 and p.file_type='predict' and d.file_type != 'predict' and p.train_id="+rid+" \
    order by p.processed_date ;"
    print "sql=",sql
    #ret=query_db(sql,db_file)
    tuple_arr=exec_sqlite.query_db(sql)
    del_files(tuple_arr, tgt_dir,tgt_patterns,1,debug)
    
# clean up old prediction files based on flag status_code != -1   ==================================== 
def del_by_day(tgt_dir,tgt_patterns, mday, db_file,debug):
    now = datetime.datetime.now()
    date_N_days_ago=now - datetime.timedelta(days=int(mday))
    mm=date_N_days_ago.month
    if mm <10:
        mm="0"+str(mm)
    else:
        mm=str(mm)
    dd=date_N_days_ago.day
    if dd <10:
        dd="0"+str(dd)
    else:
        dd=str(dd)
    str_date=str(date_N_days_ago.year)+"-"+mm+"-"+dd
    sql=" select p.id,p.file_type,p.processed_date from atdml_document p join atdml_document d on p.train_id=d.id \
    where p.train_id >0 and p.file_type='predict' and d.file_type != 'predict'  \
    and (p.processed_date < '"+str_date+"' or ( p.processed_date is null and p.created_date < '"+str_date+"') ) \
    and p.status_code != '-1' \
    order by p.processed_date ;"#limit 100000
    print "sql=",sql
    #ret=query_db(sql,db_file)
    tuple_arr=exec_sqlite.query_db(sql)
    print "sql count=",len(tuple_arr)
    del_files(tuple_arr, tgt_dir,tgt_patterns,1,debug)

# delete files by patterns, also update db: status_code = -1   ==================================== 
def del_files(tuple_arr, tgt_dir, tgt_patterns, update_db=1,debug="Y"):
    count=0
    # for each id
    for r in tuple_arr:
        id=str(r[0]) # 1st tuple element is id
        print "id=",id
        count=count+1
        # for each pattern
        for p in tgt_patterns.split(','):
            # delete file log/<id>predict.log
            #filename=os.path.join(tgt_dir,p.replace("*",id))
            #print "tgt=",filename
            fname=os.path.basename(p)
            dname=os.path.dirname(p)
            fullname=os.path.join(tgt_dir,dname,fname.replace("*",id))
            print "fullname=",fullname
            list=glob.glob(fullname)
            # for each file found
            for f in list:
                if debug=="N":
                    delete_a_file_or_dir(f)
                else:
                    print "rm",f
                        #print "tbd=",f

        #update db
        if update_db==1 and debug=="N":
            usql="update atdml_document set status_code=-1 where id="+id+";"
            uret=exec_sqlite.exec_sql(usql)  

    print "target count=",count
        
        # delete folder tmpdata/<id>retrieve
        
    return None

# delete a file or directory    
def delete_a_file_or_dir(filename,debug="N"):
    try:
        if os.path.isfile(filename):
            if debug=="N":
                os.remove(filename)
                print "file ["+filename+"] removed"
            else:
                print "rm ["+filename+"] "
        else:
            if debug=="N":
                shutil.rmtree(filename)
                print "dir ["+filename+"] removed"
            else:
                print "rmdir ["+filename+"] "
        # clean up hdfs?
    except IOError as e:
        print "Warning({0}): {1}".format(e.errno, e.strerror)
    except:
        print "Warning:", sys.exc_info()[0]  
    
'''
# query sqlite3 and return a list of tuple
def query_db(sql,db_file):
    #db_file="../../db.sqlite3"
    print "Get dictionay from db '"+db_file+"'"
    ret=None
    try:
        conn = sqlite3.connect(db_file)
        #print "Opened database successfully";
        cursor= conn.cursor()
        ret=cursor.execute(sql).fetchall() # ret a list of tuple
        
        #create dictionary
        #for record in cursor: # record is tuple        
            #print "record=",record, " t=", type(record)

        cursor.close()
        conn.close()
    except:
        print "Db retrieval error! ", sys.exc_info()
        if conn:
            conn.close()
    return ret
'''
if __name__ == '__main__':
    __description__ = "utilties for ml"
    main()