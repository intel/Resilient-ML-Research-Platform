'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''
#! /usr/bin/python

# python libraries
import os
import os.path
import sys, ConfigParser
from argparse import ArgumentParser

#####import for django database####
import sqlite3

CONF_FILE='../../app.config' # at the base dir of the web
config=ConfigParser.ConfigParser()
config.read(CONF_FILE)
DEFAULT_SQLITE=config.get('app', 'DEFAULT_SQLITE') #"/home/django/myml/db.sqlite3"

def main():
    parser = ArgumentParser(description=__description__)
    parser.add_argument("-d", "--sqlite_file", type=str, metavar="full path to sqlite file", help="full path to sqlite file", required=False)
    parser.add_argument("-s", "--str_sql", type=str, metavar="sql command", help="sql command string", required=False)
    args = parser.parse_args()

    if args.sqlite_file:
        sqlite_file = args.sqlite_file
    else:
        sqlite_file  = DEFAULT_SQLITE
    if args.str_sql:
        str_sql = args.str_sql
    else:
        str_sql  = None
        
    #print "INFO: sqlite_file=",sqlite_file
    #print "INFO: str_sql=",str_sql

    # call func
    return exec_sql(str_sql,sqlite_file)
    
def exec_sql(str_sql, sqlite_file=DEFAULT_SQLITE ):
    ret=-1
    #print "INFO: Update db '"+sqlite_file+"'"
    if str_sql:
        try:
            conn = sqlite3.connect(sqlite_file)
            #print "Opened database successfully";
            conn.execute(str_sql)
            conn.commit()
            ret=conn.total_changes
            conn.close()
            #print 'Done DB!'
        except:
            print "ERROR: Db update error! ", sys.exc_info()
            if conn:
                conn.close()
                   
    else:
        print "ERROR: Sql command error. Sql=", str_sql
        
    return ret

# return dictionary from feature importance
def get_dict(rid, sqlite_file=DEFAULT_SQLITE ):
    all_verified=dict()
    human_verified=dict()
    #print "INFO: Get dictionay from db '"+sqlite_file+"'"

    try:
        conn = sqlite3.connect(sqlite_file)
        #print "Opened database successfully";
        cursor= conn.cursor()
        cursor.execute("select fid, vote from atdml_feature_click " \
                +" where rid="+rid+" and vote >0 ")
        
        #create dictionary
        for record in cursor:
            all_verified[str(record[0])]=record[1]
            if record[1] > 5:
                human_verified[str(record[0])]=record[1]
        cursor.close()
        conn.close()
    except:
        print "ERROR: Db retrival error! ", sys.exc_info()
        if conn:
            conn.close()
    return all_verified, human_verified     

# query sqlite3 and return a list of tuple ========================
def query_db(sql, sqlite_file=DEFAULT_SQLITE):
    #sqlite_file="../../db.sqlite3"
    #print "INFO: Query db file='"+sqlite_file+"'"
    ret=None
    try:
        conn = sqlite3.connect(sqlite_file)
        cursor= conn.cursor()
        ret=cursor.execute(sql).fetchall() # return a list of tuple

        cursor.close()
        conn.close()
    except:
        print "ERROR: Db retrieval error! ", sys.exc_info()
        if conn:
            conn.close()
    return ret    
    
if __name__ == '__main__':
    __description__ = "update sqlite for web"
    main()
