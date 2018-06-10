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
import mysql.connector
import sqlite3
from argparse import ArgumentParser

CONF_FILE='../../app.config' # at the base dir of the web
config=ConfigParser.ConfigParser()
config.read(CONF_FILE)
# the flag to use sqlite or mysql if empty
DEFAULT_SQLITE=config.get('app', 'DEFAULT_SQLITE') #"/home/django/myml/db.sqlite3"
conn_params={}

def main():
    parser = ArgumentParser(description=__description__)
    parser.add_argument("-d", "--sqlite_file", type=str, metavar="full path to sqlite file", help="full path to sqlite file", required=False)
    parser.add_argument("-s", "--str_sql", type=str, metavar="sql command", help="sql command string", required=False)
    args = parser.parse_args()

    sqlite_file=None
    if args.sqlite_file:
        sqlite_file = args.sqlite_file
    else:
        sqlite_file  = DEFAULT_SQLITE
    if args.str_sql:
        str_sql = args.str_sql
    else:
        str_sql  = None
    
    if sqlite_file == "" or sqlite_file is None:
        conn_params["host"]=config.get('mysql', 'ip_address')
        conn_params["port"]=config.get('mysql', 'port')
        conn_params["db"]=config.get('mysql', 'db_name')
        conn_params["username"]=config.get('mysql', 'username')
        conn_params["password"]=config.get('mysql', 'password')
        
    #print "INFO: sqlite_file=",sqlite_file
    #print "INFO: str_sql=",str_sql
    #print "INFO: conn_params=",conn_params

    conn, type=get_conn(sqlite_file,conn_params)
    ret=0
    #print "conn=",conn,",type=",type
    # call func
    ret=exec_sql(str_sql,conn, type)
    #ret=query_db(str_sql,conn, type)
    #a,b=get_dict("3",conn, type)
    #print "a=",a,",b=",b
    #print "ret=",ret
    return ret

# get connection: return conn and type
def get_conn(sqlite_file=DEFAULT_SQLITE, in_params=None):
    #print "XXXXXXXXXXXXXXXXXX in get_conn()"
    if sqlite_file == "" or sqlite_file is None:
        if in_params is None:
            in_params={}
            in_params["host"]=config.get('mysql', 'ip_address')
            in_params["port"]=config.get('mysql', 'port')
            in_params["db"]=config.get('mysql', 'db_name')
            in_params["username"]=config.get('mysql', 'username')
            in_params["password"]=config.get('mysql', 'password')
        if in_params is None or len(in_params)==0:
            return None
    # for sqlite
    if sqlite_file and sqlite_file>"":
        try:
            conn = sqlite3.connect(sqlite_file)
            return conn, "sqlite"
        except:
            print "ERROR: Sqlite db connection error! ", sys.exc_info()
            if conn:
                conn.close()
    else:
        try:
            #print "in_params=",in_params
            conn = mysql.connector.connect(
                user=in_params["username"], password=in_params["password"]
                , host=in_params["host"], port=in_params["port"], database=in_params["db"])
            return conn, "mysql"
        except:
            print "ERROR: Mysql db connection error! ", sys.exc_info()
            if conn:
                conn.close()
    return None, None
    
# mysql: execute the str_sql
def mysql_exec_sql(str_sql, conn):
    ret=-1
    try:
        cursor = conn.cursor()
        cursor.execute(str_sql)
        conn.commit()
        ret=cursor.rowcount
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    except:
        print "ERROR: Db update error! ", sys.exc_info()
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        return None
        
    return ret
# sqlite: execute the str_sql    
def exec_sql(str_sql , conn=None, type=None):
    ret=-1
    if conn is None:
        conn, type =get_conn()
    
    if type == "mysql":
        return mysql_exec_sql(str_sql,conn)
        
    #print "INFO: Update db '"+sqlite_file+"'"
    if str_sql:
        try:
            #conn = sqlite3.connect(sqlite_file)
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

# mysql: return dictionary from feature importance========================
def mysql_get_dict(rid, conn=None, type=None):
    #print "in mysql_get_dict"
    ret=-1
    all_verified=dict()
    human_verified=dict()
    try:
        cursor = conn.cursor()
        cursor.execute("select fid, vote from atdml_feature_click " \
                +" where rid='"+str(int(rid))+"' and vote >0;")
        #ret=cursor.rowcount
        #create dictionary
        
        for record in cursor:
            #print "record=",record
            all_verified[str(record[0])]=record[1]
            if record[1] > 5:
                human_verified[str(record[0])]=record[1]

        if cursor:
            cursor.close()
        if conn:
            conn.close()
    except:
        print "ERROR: Db get dict error! ", sys.exc_info()
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        return None, None
        
    return all_verified, human_verified
# sqlite: return dictionary from feature importance
def get_dict(rid , conn=None, type=None):
    if conn is None:
        conn, type =get_conn()
    #print "in get_dict; type=",type    
    if type == "mysql":
        return mysql_get_dict(rid,conn)


    all_verified=dict()
    human_verified=dict()
    print "INFO: Get dictionay from db rid='"+str(rid)+"'"

    try:
        #conn = sqlite3.connect(sqlite_file)
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

# mysql: return a list of tuple ========================
def mysql_query_db(sql, conn=None, type=None):
    ret=-1
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows=cursor.fetchall() # return a list of tuple
        ret=len(rows)
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    except:
        print "ERROR: Db query error! ", sys.exc_info()
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        return None
        
    return ret    
# sqlite: query sqlite3 and return a list of tuple ========================
def query_db(sql, conn=None, type=None):
    if conn is None:
        conn, type =get_conn()

    if type == "mysql":
        return mysql_query_db(sql,conn)

        #sqlite_file="../../db.sqlite3"
    #print "INFO: Query db file='"+sqlite_file+"'"
    ret=None
    try:
        #conn = sqlite3.connect(sqlite_file)
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
    __description__ = "update db for web"
    main()
