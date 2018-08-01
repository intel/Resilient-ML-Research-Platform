#! /usr/bin/python
'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''
# standard library imports
from argparse import ArgumentParser
import sys, os, json
import gzip
import pandas as pd
from pandas import read_csv
import datetime
from datetime import datetime as dt
import random, string


def arg_parser(parser):
    # input output file info
    parser.add_argument("-fn", "--fname", type=str, metavar="name of data file", help="name of data file", required=False)
    parser.add_argument("-ofn", "--ofname", type=str, metavar="name of output file", help="name of output file", required=False)
    #dataset info
    parser.add_argument("-r", "--row_id", type=str, metavar="row id", help="row_id number in the db", required=False)
    parser.add_argument("-pm", "--params", type=str, metavar="json string", help="json string of parameters", required=False)
    
    parser.add_argument("-ss", "--sep_snt", type=str, metavar="separator for sentence", help="separator for sentence", required=False)
    parser.add_argument("-sf", "--start_flag", type=str, metavar="str to get data after", help="get data after this string", required=False)
    parser.add_argument("-tl", "--time_list", type=str, metavar="time series list in sec", help="time series list in sec", required=False)
    parser.add_argument("-md", "--mode", type=str, metavar="mode of this app", help="mode: csv", required=False)
    
    return parser.parse_args()
  
'''
python bin/cf_csv.py 
'''  
def main():  # ============= =============  ============= =============
    # parse arguments
    parser = ArgumentParser(description=__description__)
    args = arg_parser(parser)

    if args.fname:
        fname = args.fname
    else:
        fname  = '/mnt/32t/etl/vipermonitor_dirty/0423_kaspersky_p20s3/00b4d8bf603522c86b572819beac6d7c55ded1800368071fe74ed3a280e2ca45.procmon_date_.r02w81.gz'
    if args.ofname:
        ofname = args.ofname
    else:
        ofname  = None
    if args.sep_snt:
        sep_snt = args.sep_snt
    else:
        sep_snt  = '\t'
    if args.start_flag:
        start_flag = str(args.start_flag)
    else:
        start_flag  = None
    if args.time_list:
        time_list = eval(args.time_list)
    else:
        time_list  = None
    if args.mode:
        mode = args.mode
    else:
        mode  = None
        
        
    if args.row_id:
        row_id_str = str(args.row_id)
    else:
        row_id_str  = "0"
    params=None
    if args.params:
        params=args.params
    
    ## generate sub samples based on time list
    if mode == 'csv_ts':
        df=open_df(fname,header=None,sep=",",quotechar='"')
        # find start flag
        if start_flag:
            df=find_start_row(df,path_field=3)
        # cleanup special char
        df=cleanup_csv(df)
        # get the first part of name as output fname
        base_name=os.path.basename(fname).split('.')[0]
        # generate sub samples based on time list
        return get_time_series(df, time_field=0, time_list=time_list, out_fname=base_name)
    else:
        # for fsw
        #print ("INFO: start_flagxx=",start_flag)
        return to_one_line_fsw(row_id_str, params, fname, ofname, sep_snt, start_flag
            , time_list=time_list)

# Convert csv dataset file to libsvm directly            
# Input: 
#    line string: "<label>,<feat0>,<feat1>,<feat2>,..."
# Return:
#    
def featuring(line, featuring_params):
    jparams={}
    ret_arr=[]
    custom=None
    #
    if featuring_params and len(featuring_params)>0:
        try:
            jparams=json.loads(featuring_params)
        except Exception as e:
                print "ERROR: user module error.", e.__doc__, e.message
                return -200
    #
    if 'custom' in jparams:
        custom=jparams['custom']
    if custom is None:
        custom='csv'

    # chk empty 
    if not line or len(line)==0:
        return None
    #
    if custom == 'csv':
        return list2libsvm(line,jparams)
       

    #str_lines=line.split(delimitor)
    return None

# convert list to libsvm, index starting by 1, ignore None/Null value
#def list2libsvm(in_list, label_index=0, add_meta=True, label_dict=None):
def list2libsvm(line, jparams, add_meta=True):
    out=""
    delm=""
    meta=""
    idx=1
    separator=','
    label_index=None
    label_dict=None
    if line is None or len(line)==0:
        return None
    
    if 'label_index' in jparams:
        try:
            if isinstance(jparams['label_index'],int):
                label_index=jparams['label_index']
            else:
                label_index=eval(jparams['label_index'])
        except:
            label_index=None
    if 'label_dict' in jparams:
        label_dict=jparams['label_dict']
    if 'separator' in jparams:
        separator=jparams['separator']
        
    # get as list
    if isinstance(line, list):
        in_list=line
    else:
        in_list=line.split(separator)
        
    for i,v in enumerate(in_list):
        if v:
            if label_index == i: #label
                if label_dict:
                    meta=str(label_dict[v])+" "
                else:
                    meta=str(v)+" "
            else:
                out=out+delm+str(idx)+":"+str(v)
                delm=" "
                idx+=1
    # calculate hash for all data
    meta=str(djb2_(out))+" "+meta
    return meta+out    
            
# ================================================================================== to_one_line_fsw() ============ 
# filter/clean data and convert a csv file (a sample) to one line string
#  filter rows by filter_list & path_filters;
# Input: filename 
# Return a string, to stdout or to ofname 
def to_one_line_fsw(fname=None, params=None,  ofname=None, sep_snt='\t', start_flag=None
        , filter_list=None, filter_field_name=None, path_filters=None
        , usecols=None, col_names=None
        , header=None, sep=',',quotechar='"' , time_list=None, printout=True):
        
    # get default parameters
    jparams=get_params(params)
    path_field_name=None
    # set params
    #  for filter
    if 'filter_list' in jparams:
        filter_list=jparams['filter_list']
    if 'filter_field_name' in jparams:
        filter_field_name=jparams['filter_field_name']
    if 'path_field_name' in jparams:
        path_field_name=jparams['path_field_name']
    if 'time_field_name' in jparams:
        time_field_name=jparams['time_field_name']
    if 'path_filters' in jparams:
        path_filters=jparams['path_filters']
    #   find start flag    
    if 'start_flag' in jparams:
        start_flag=jparams['start_flag']

    
    # for loading CSV file
    if 'usecols' in jparams:
        usecols=jparams['usecols']
    if 'col_names' in jparams:
        col_names=jparams['col_names']
    if 'header' in jparams:
        header=jparams['header']
    meta_data=['','','']
    if 'meta_data' in jparams:
        meta_data=jparams['meta_data']
        
    if 'sep' in jparams:
        sep=jparams['sep']
    if 'quotechar' in jparams:
        quotechar=jparams['quotechar']
    # input fanme
    if 'fname' in jparams and fname is None:
        fname=jparams['fname']
    # outpur fname
    if 'ofname' in jparams and ofname is None:
        ofname=jparams['ofname']
    
    #print("fname=",fname )    
    # Load gz file to Pandas DataFrame =======================
    if fname.endswith('.gz'):
        df = pd.read_csv(fname, compression='gzip', header=header, sep=sep, quotechar=quotechar
            , usecols=usecols, names=col_names, error_bad_lines=False)
    else: # no compression
        df = pd.read_csv(fname, header=header, sep=sep, quotechar=quotechar
            , usecols=usecols, names=col_names, error_bad_lines=False)
    #print("columns=",df.columns.values )
    #print("df=",df.head(2) )
    #print("df s=",df.shape )

    # Find start flag "start_" and exclude data before  =======================
    if path_field_name in df.columns and start_flag and len(start_flag)>0:
        ix=df[df[path_field_name].str.contains(start_flag)].index.tolist()[-1]+1
        #print ("INFO: start_ ix=",ix)
        # get data after ix, if start_ exist
        if ix>0:
            df=df.iloc[ix:]
            #print("df=",df.head(5))
        else:
            sys.stderr.write("INFO: start_ flag not found\n")

    #print("df s=",df.shape )
    # Clean up special chars for in each cell from CSV file
    #  replace all , to _ ; remove non-printable char  =======================
    df=df.applymap(lambda x: str(x).replace(',','_').replace('\n',';').replace('\t',' ;').replace(' ','_').replace('"','')   ) \
         .applymap(lambda x: ''.join(["" if ord(i) < 32 or ord(i) > 126 else i for i in x]))
         
         
    # Filter a field  ====================================== ================
    if filter_field_name in df.columns and filter_list and len(filter_list)>0:
        #Filter notepad04_word.exe|timeout.exe|shut2down.exe|ftp.exe|WINWORD.EXE|screenCapture.exe
        df=df[~df[filter_field_name].isin(filter_list)]

    #print("df s=",df.shape )
    odf=df
    #print("odf s=",odf.shape )
    # split and filter by path
    if path_field_name in df.columns:
        odf=filter_parse_path(df, path_field_name=path_field_name, drop_flag=True
            , path_filters=path_filters, replacement_list=['ami_runner1','ami'])
    #print("odf s=",odf.shape )
    odf_list=None
    if time_list and len(time_list)>0 and time_field_name in df.columns:
        odf_list=get_time_series(odf, time_field_name, time_list=time_list)
    
    # generate time list data file
    if odf_list and len(odf_list)>0:
        hash=meta_data[1]
        for i, ts in enumerate(odf_list):
            ts=ts.applymap(lambda x: str(x))
            meta_data[1]=hash+"-"+str(time_list[i]).zfill(5)
            all_arr=[' '.join(ln) for ln in ts.values]
        #print sep_snt.join(meta_data)+sep_snt+sep_snt.join(all_arr)
        meta_data[1]=hash+"-all"

    # output to CSV file  =======================
    # concat all cells to string   
    if ofname and len(ofname)>0:
        # not used ...
        odf.to_csv(ofname)
    else: # echo to stdout
        odf=odf.applymap(lambda x: str(x))
        all_arr=[' '.join(ln) for ln in odf.values]
        out=sep_snt.join(meta_data)+sep_snt+sep_snt.join(all_arr)
        if printout:
            print out
        else:
            return out
    
    return 


# ================================================================================== to_one_line() ============ 
# Convert csv to 
#  assume csv file has title with one line of data
# Input: filename 
# Return:
#   
def to_one_line(fname, params=None):
    # get default parameters
    jparams=json.loads(params)
    path_field_name=None
    
    has_header=None
    usecols=None
    col_names=None
    meta_data=['','','']
    sep=","
    quotechar='"'
    # for loading CSV file
    if 'usecols' in jparams:
        usecols=jparams['usecols']
    if 'col_names' in jparams:
        col_names=jparams['col_names']
    if 'has_header' in jparams:
        header=eval(jparams['has_header'])
        if header:
            header=0
            
    if "label_index" in jparams:
        label_index=jparams['label_index']
        
    if 'sep' in jparams:
        sep=jparams['sep']
    if 'quotechar' in jparams:
        quotechar=jparams['quotechar']

    # Load gz file to Pandas DataFrame =======================
    if fname.endswith('.gz'):
        df = pd.read_csv(fname, compression='gzip', header=header, sep=sep, quotechar=quotechar
            , usecols=usecols, names=col_names, error_bad_lines=False)
    else: # no compression
        df = pd.read_csv(fname, header=header, sep=sep, quotechar=quotechar
            , usecols=usecols, names=col_names, error_bad_lines=False)    
    
    row_str=df.to_string(header=False,index=False,index_names=False) 
    #print df.head() 
    #print "row_str=",row_str
    # remove space
    line=sep.join(row_str.split())
    # get 1st item
    ret_str=list2libsvm(line, jparams)

    #print "jparams=",jparams
    #print "line=",line
    #print "ret_str=",ret_str
    return ret_str

# ================================================================================== cleanup_csv ============ 
# Clean up special chars for in each cell from CSV file  
def cleanup_csv(df):    
    # 
    #  replace all , to _ ; remove non-printable char  =======================
    df=df.applymap(lambda x: str(x).replace(',','_').replace('\n',';').replace('\t',' ;').replace(' ','_').replace('"','')   ) \
         .applymap(lambda x: ''.join(["" if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    return df
    
# ================================================================================== open_df() ============ 
# Find start flag "start_" and exclude data before  
def find_start_row(df,path_field='path',start_flag="start_"):    
    
    # Find start flag "start_" and exclude data before  =======================
    if isinstance(path_field, int):
        ix=df[df[path_field].str.contains(start_flag)].index.tolist()[-1]+1
    elif path_field_name in df.columns and start_flag and len(start_flag)>0:
        ix=df[df[path_field_name].str.contains(start_flag)].index.tolist()[-1]+1
        #print ("INFO: start_ ix=",ix)
        # get data after ix, if start_ exist
        
    if ix>0:
        df=df.iloc[ix:]
        #print("df=",df.head(5))
    else:
        sys.stderr.write("INFO: start_ flag not found\n")
    return df
    

# ================================================================================== open_df() ============ 
# open dataframe from csv
def open_df(fname,header,sep=",",quotechar='"', usecols=None, names=None,error_bad_lines=False):    
    if fname.endswith('.gz'):
        df = pd.read_csv(fname, compression='gzip', header=header, sep=sep, quotechar='"'
            , usecols=usecols, names=names, error_bad_lines=error_bad_lines)
    else: # no compression
        df = pd.read_csv(fname, header=header, sep=sep, quotechar='"'
            , usecols=usecols, names=names, error_bad_lines=error_bad_lines)
    return df
    
# ================================================================================== time series () ============ 
# fitler path and split to get last dir  and filename parts  
def get_time_series(df, time_field, time_list=None, tformat='%Y-%m-%dT%H:%M:%S', out_fname=None):
    
    # base time by name or index number
    start_time_str=df.iloc[0][time_field]
    print "start_time_str=",start_time_str
    #start_time=dt.strptime('2018-04-09T22:32:27','%Y-%m-%dT%H:%M:%S')
    
    ret=[]
    start_time=dt.strptime(start_time_str,tformat)
    if time_list:
        for i in time_list:
            print "time at=",i
            sec_delta=datetime.timedelta(seconds=i)
            stop_time=start_time+sec_delta
            oret=df[df[time_field]<=dt.strftime(stop_time,tformat)]
            ret.append(oret)
            #print("stop_time=",stop_time,"len=",len(oret))

        # get time from last row ?
        #ltime_str=df.ix[-1:][time_field_name]
        #last_time=dt.strptime(ltime_str,tformat)
    if out_fname:
        for i,v in enumerate(time_list):
            print "df at=",i
            ofname=out_fname+"_"+str(v).zfill(3)+".csv"
            df=ret[i]
            df.to_csv(ofname, sep=',', index= False, header=False)
        return 0
    else:
        return ret
    
# ================================================================================== filter_parse_path () ============ 
# fitler path and split to get last dir  and filename parts  
def filter_parse_path(df, path_field_name="path", drop_flag=True, path_filters=None, replacement_list=None):    
    odf=df
    #print("titles=",list(df), type(df))
    #print("df=",df.head(5))
    #print("df s=",df.shape, "odf s=",odf.shape)

    
    # split PATH to get last 2 parts:  dir and fname parts ======================
    df['dir_part'],df['fname_part']=df[[path_field_name]].applymap(lambda x: ' '.join(x.replace(' ','_').split('\\')[-2:]))[path_field_name].str.split(' ',1).str

    # remove path col 
    if drop_flag:
        odf=df.drop([path_field_name],axis=1)
    #print("odf=",odf.head(5)) 
    #print("odf=",odf.shape) 
    
    #Filter dir and filename  =======================
    if path_filters and len(path_filters)>0:
        odf=odf[~odf['dir_part'].isin(path_filters)]
        odf=odf[~odf['fname_part'].isin(path_filters)]
    #print("odf=",odf.shape) 
    #print("fdf=",odf.head(5))
    
    # Replace ami or ami_runner1 by random strings in both dir and file names
    if replacement_list and len(replacement_list)>0:
        for str_from in replacement_list:
            #print("repl=",str_from)
            # random string to replace testing account "ami"  =======================
            str_to=''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
            # string replace
            odf['dir_part']=odf['dir_part'].replace(str_from,str_to)
            odf['fname_part']=odf['fname_part'].replace(str_from,str_to)
    return odf
# ================================================================================== get_params () ============ 
# set default values    
def get_params(str_params):
    # get parameters
    jparams={}
    try:
        if str_params and len(str_params)>0:
            jparams = json.loads(str_params)
        # set default
        if 'type' in jparams:
            # =========================================== procmon ==========
            if jparams['type']=='procmon':
                if not 'filter_field_name' in jparams:
                    jparams['filter_field_name']='proc'
                if not 'path_field_name' in jparams:
                    jparams['path_field_name']='path'
                    
                if not 'filter_list' in jparams:
                    jparams['filter_list']=['notepad04_word.exe','timeout.exe','shut2down.exe','ftp.exe','WINWORD.EXE','screenCapture.exe']
                if not 'path_filters' in jparams:
                    jparams['path_filters']=['runas.exe','get_and_run_procmom.bat','TBO_myDoc2.docx','shut2down.exe'
                        ,'lab_ftp','simulate_user_actions.bat','out_ftp_put_screen.logx','takeScreenShot.bat','screenshot.bat'
                        ,'start_clean_ftp.bat','shutdown_backup.bat','Procmon64.exe']
                if not 'usecols' in jparams:
                    #jparams['usecols']=[0,2,4,13,14,3,1] didn't work
                    jparams['usecols']=[0,1,2,3,4,13,14]
                # "0Time of Day","1Process Name","2PID","3Operation","4Path"
                #  ,"5Result","6Detail",7"Date & Time","8Image Path","9Company","10Description"
                #  ,"11User","12Command Line","13TID","14Parent PID"
                if not 'col_names' in jparams:
                    #jparams['col_names']=['time','pid','path','tid','ppid','op','proc']
                    jparams['col_names']=['time','proc','pid','op','path','tid','ppid']
            # =========================================== fsw ==========
            
            elif jparams['type']=='fsw':
                if not 'usecols' in jparams:
                    jparams['usecols']=[0,1,2,3,4,5]
                if not 'path_field_name' in jparams:
                    jparams['path_field_name']='path'                    
                if not 'time_field_name' in jparams:
                    jparams['time_field_name']='time'                    
                if not 'col_names' in jparams:
                    jparams['col_names']=["time","pid","op","path","entropy","magicbyte"]
                if not 'path_filters' in jparams:
                    jparams['path_filters']=['runas.exe','get_and_run_procmom.bat','TBO_myDoc2.docx','shut2down.exe'
                        ,'lab_ftp','simulate_user_actions.bat','out_ftp_put_screen.logx','takeScreenShot.bat','screenshot.bat'
                        ,'start_clean_ftp.bat','shutdown_backup.bat','Procmon64.exe']
             
        # for loading CSV file
        #if not 'usecols' in jparams:
        #    jparams['usecols']=[1,3,4]
        #if not 'col_names' in jparams:
        #    jparams['col_names']=['proc','op','path']

    except Exception, e:
        #e = sys.exc_info()[0]
        print("WARNING: error in loading JSON jparams.", str(e))
        
    return jparams
    
# string hash function   ============= =============
def djb2_(key, max_feat_cnt=1442968193):
    hash = 5381 
    for k in key:
        hash = ((hash << 5) + hash) + ord(k) 
    hash_ret = hash % max_feat_cnt
    return hash_ret
    
if __name__ == '__main__':
    __description__ = "preprocess procmon data"
    main()