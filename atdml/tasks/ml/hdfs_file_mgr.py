#! /usr/bin/python
'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''

from argparse import ArgumentParser
import sys, ConfigParser
import datetime, time
import os, shutil, glob, re, json, gzip
import pydoop.hdfs as hdfs
FILES_PER_GZ=2000

def main():
    parser = ArgumentParser(description=__description__)
    parser.add_argument("-r", "--row_id", type=str, help="row_id number in the db"
        , dest='rid', default ="0", required=False)
    parser.add_argument('-act', '--action', type = str, help = 'action applied to target files'
        , dest='action', default ="upsert_a_file", required =False)

    parser.add_argument('-sdir', '--src_dir', type = str, help = 'directory of source files'
        , dest='src_dir', required =False)
    parser.add_argument('-tdir', '--hdfs_tgt_dir', type = str, help = 'target hdfs directory'
        , dest='hdfs_tgt_dir', required =False)
    parser.add_argument('-ddir', '--dest_dir', type = str, help = 'destination directory in local file system'
    , dest='dest_dir', required =False)
    
        
    parser.add_argument('-fn', '--filname', type = str, help = 'target filename'
        , dest='filename', default =None, required =False)
    parser.add_argument('-lb', '--label', type = str, help = 'label for the folder'
        , dest='label', default =None, required =False)
    parser.add_argument('-lbc', '--label_cutoff', type = str, help = 'label for the folder'
        , dest='label_cutoff', default =20, required =False)
    parser.add_argument('-afn', '--add_family_name', type = str, help = 'append family name to info'
        , dest='add_family_name', default =None, required =False)

    parser.add_argument('-dg', '--debug', type = str, help = 'print file list only'
        , dest='debug', default ="Y", required =False)
    args = parser.parse_args()

    t0 = time.time()
    if args.action:
        print "in action=" ,args.action,",src_dir=",args.src_dir,",hdfs_tgt_dir=",args.hdfs_tgt_dir

        if args.action == "upsert_a_file": #  python ./ml/file_mgr.py -act del_by_day -md 14 -pttn "log/*predict.log,tmpdata/*retrieve" -dg Y 
            upsert_a_file(args.src_dir, args.hdfs_tgt_dir, args.filename, args.debug)
        elif args.action == "upsert_a_folder":  
            upsert_a_folder(args.src_dir, args.hdfs_tgt_dir, args.filename, args.debug)
        elif args.action == "etl_android_folder":  
            etl_android_folder(args.src_dir, args.hdfs_tgt_dir, args.filename, args.label, args.debug)
        elif args.action == "ex_gz_folder":
            ex_gz_folder(args.src_dir, args.filename, args.label, add_family_name=args.add_family_name, dest_dir=args.dest_dir)
        elif args.action == "etl_family_folders":
            etl_family_folders(args.src_dir, args.filename, args.label, add_family_name=args.add_family_name, dest_dir=args.dest_dir)
        elif args.action == "combine_gz_folder":
            combine_gz_folder(args.src_dir, args.filename,args.label_cutoff)
    t1 = time.time()
    print 'INFO: running time: %f' %(t1-t0)

            
'''
    # need to set mapred-site.xml to inatll pydoop
    export HADOOP_USER_NAME=hadoop
    export JAVA_HOME=/usr/lib/jvm/java
    export HADOOP_HOME=/home/android/Downloads/hadoop_latest
    export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
    export PATH=$PATH:$HADOOP_HOME/sbin:$HADOOP_HOME/bin:$JAVA_HOME/bin

 python ./hdfs_file_mgr.py -sdir '/nfs/output/archives/100k-20160511/xposed/clean' 
 -fn '*.only.log' -act "ex_gz_folder"
 python ml/hdfs_file_mgr.py -sdir '/home/django/test_data' -tdir '/user/hadoop/upload/data_retrieved/android04' 
 -fn '*.gz' -act "upsert_a_folder" -dg N
 
 python hdfs_file_mgr.py -act ex_gz_folder -afn Y -sdir "/mnt/A5T01/samples/dynamic_logs/result/hummingBxx" -fn "*.only.log.list" -ddir . -lb "dirty"
  python hdfs_file_mgr.py -act etl_family_folders -afn Y -sdir "/mnt/A5T01/samples/dynamic_logs/result/*" -fn "*.only.log.list" -ddir . -lb "dirty"
 '''
# open libsvm file, replace label and output to .gz file
def combine_gz_folder(src_dir, filename, label_cutoff=20, fraction=5):
    src_fname=os.path.join(src_dir,filename)
    # get source info
    src_fs=glob.glob(src_fname)
    # create .gz file for output
    gz_filename=os.path.join(src_dir,"output"+".gz")
    ret=os.system("echo -n '' | gzip -9 > "+gz_filename)
    print "gz_filename=",gz_filename
    f_out=gzip.open(gz_filename, 'wb')
    
    f_t_out=None
    if fraction > 0:
        gz_t_filename=os.path.join(src_dir,"output_test"+".gz")
        ret=os.system("echo -n '' | gzip -9 > "+gz_t_filename)
        print "gz_t_filename=",gz_t_filename
        f_t_out=gzip.open(gz_t_filename, 'wb')
    
    
    in_gz_cnt=0
    label_cutoff=int(label_cutoff)
    for sf in src_fs:
        in_gz_cnt=in_gz_cnt+1
        print in_gz_cnt," =",sf, "=", ",label_cutoff=",label_cutoff
        #with gzip.open(sf, 'rb') as f_in: 
        l_count=0
        with open(sf, 'rb') as f_in: 
            for line in f_in:
                l_count=l_count+1
                token=line.split(' ')
                #print "token=",int(token[0]),",line=",line
                
                # set all label >  cutoff with same label for testing
                if int(token[0]) > label_cutoff:
                    #print "before=", line
                    rline=line.replace(token[0],str(label_cutoff),1)
                    #print "after=",line
                    f_t_out.write(line) 
                    f_out.write(rline)
                elif int(token[0]) == label_cutoff:
                    rline=line.replace(token[0],str(label_cutoff*-1),1)
                    f_t_out.write(rline) # neg for real label same as cutoff label
                    f_out.write(line)
                elif fraction > 0 and l_count % fraction ==0 :
                    #print "in!!"
                    rline=line.replace(token[0],str(label_cutoff),1)
                    f_t_out.write(line) 
                    f_out.write(rline)
                else:
                    f_out.write(line)
 
# parse local files in a folder to gz files
# assume the basename of src_dir is label/family name, filename has wildcard * for all similar files
# use shell commands to speed up process 
def ex_gz_folder(src_dir, filename, label, add_family_name="N", dest_dir=None, sub_dir=None, gz_fname_counter=0):
    
    # get source info
    if not sub_dir is None and len(sub_dir)>0:
        src_fname=os.path.join(src_dir,sub_dir,filename)
    else:
        src_fname=os.path.join(src_dir,filename)
    print "src_fname=",src_fname
    
    # get list from wildcard
    src_fs=glob.glob(src_fname)

    # get label from folder
    if label is None:
        label=os.path.basename(src_dir)
    family_name=""
    if add_family_name=="Y":
        family_name=os.path.basename(src_dir)
    print "label=",label, ",file count=", len(src_fs),",family_name=",family_name #,",src_fs=",src_fs

    # create .gz file for upload
    in_gz_cnt=0
    gz_filename=None
    for sf in src_fs:

        if len(sf) <3:
            print "-- filter by size, pass ", sf, ""
            pass

        in_gz_cnt=in_gz_cnt+1


        if in_gz_cnt % FILES_PER_GZ==1:
            gz_fname_counter=gz_fname_counter+1
            if dest_dir is None:
                gz_filename=os.path.join(src_dir,label+"_"+str(gz_fname_counter)+".gz")
            else:
                gz_filename=os.path.join(dest_dir,label+"_"+str(gz_fname_counter)+".gz")
            ret=os.system("echo -n '' | gzip -9 > "+gz_filename)
            print "gz_filename=",gz_filename

        # get hash from filename; remove filename extension
        # assume filename starting with <hash>.x.x.x
        filename=os.path.basename(sf)
        if filename.index('.')>0:
            hash_fname=filename[0:filename.index('.')]
        else:
            hash_fname=filename
            
        # add family name to info    
        if add_family_name=="Y" and len(family_name)>0:
            hash_fname=hash_fname+"."+family_name
        # echo progress   
        print "++",in_gz_cnt,hash_fname
            
        # date
        try:
            src_ctime_int=int(os.path.getctime(sf))
        except:
            src_ctime_int=None
            
        fdate=datetime.datetime.fromtimestamp(src_ctime_int).strftime("%Y-%m-%d %H:%M:%S")
        meta_fs="%s%s%s%s%s%s" % (label,"\\t",hash_fname,"\\t",fdate,"\\t")
        #print "meta_fs=",meta_fs    
        # add meta data fields, 
        try:
            
            cmd="cat "+sf \
                +" | tr '\\t' ' ' | tr '\\n' '\\t' | sed 's/\\t$/\\n/' | sed 's/^/" \
                +meta_fs+"/' | gzip -9 >> " \
                + gz_filename
            #print "cmd=", cmd
            ret=os.system(
                cmd
            )
            #print "ret=", ret
        except:
            e = sys.exc_info()[0]
            print "Error 1: ", e
    return gz_fname_counter

# assume  folder stucture ./<family>/<hash>/<hash>.only.log.list
#   src_dir has wildcard; filename has wildcard
def etl_family_folders(src_dir, filename, label, add_family_name="Y", dest_dir=None, sub_dir="*"):
            
    # get list from wildcard
    src_dirs=glob.glob(src_dir)
    #print "src_dirs=",src_dirs
    gz_fname_counter=0
    for f in sorted(src_dirs):
        if os.path.isdir(f):
            print "+++",os.path.basename(f)
            gz_fname_counter=ex_gz_folder(f, filename, label, add_family_name, dest_dir, sub_dir,gz_fname_counter)
            
            
# parse local files in a folder and upload to HDFS
def etl_android_folder(src_dir, hdfs_tgt_dir, filename, label, debug): 
    src_fname=os.path.join(src_dir,filename)
    tgt_fname=os.path.join(hdfs_tgt_dir,filename)
    
    # get source info
    src_fs=glob.glob(src_fname)
    
    # get label from folder
    if label is None:
        label=os.path.basename(src_dir)
    print "label=",label #,",src_fs=",src_fs
    
    ''' can't clean up the folder.
    # clean up/recreate target folder
    if debug=="N" and len(src_fs)>0:
        print "recreate folder", hdfs_tgt_dir
        hdfs.rmr(hdfs_tgt_dir)
        hdfs.mkdir(hdfs_tgt_dir)
    '''
    # create .gz file for upload
    in_gz_cnt=0
    gz_file=None
    gz_filename=None
    for sf in src_fs:
    
        if len(sf) <3:
            pass
            
        in_gz_cnt=in_gz_cnt+1
        
        print in_gz_cnt,"===========",sf, "==========="
        src_bfname=os.path.basename(sf)
        tgt_fname=os.path.join(hdfs_tgt_dir,src_bfname)
        
        if in_gz_cnt % FILES_PER_GZ==1:
            if not gz_file is None:
                gz_file.close()
                if not gz_filename is None:
                    upsert_a_file(src_dir, hdfs_tgt_dir, os.path.basename(gz_filename), debug)
            gz_filename=os.path.join(src_dir,label+"_"+str(in_gz_cnt)+".gz")
            print "gz_filename=",gz_filename
            gz_file=gzip.open(gz_filename,'wb')
            
        # extract, combine and upload
        try:
            #ret=process_android_jlog(src_dir, src_bfname, label)
            ret=process_text_log(src_dir, src_bfname, label)
            #jret=json.loads(ret)
            gz_file.write(ret+'\n')
            #print "ret3=", ret
        except:
            e = sys.exc_info()[0]
            print "Error 1: ", e

    if not gz_file is None:
        gz_file.close()
        if not gz_filename is None:
            upsert_a_file(src_dir, hdfs_tgt_dir, os.path.basename(gz_filename), debug)
            
# expect lines in format .*\n
# assume md5 in filename, date as mdate of log file, label from param, each line is a event/func call
#   return json string with <label>\t<md5>\t<date>\t<func call>\t...\n
def process_text_log(src_dir, filename, label=None, src_ctime_int=None, start_str="{"):
    delimiter='\t'
    src_fname=os.path.join(src_dir,filename)
    #print "src_fname=",src_fname
    lines = [line.rstrip('\n') for line in open(src_fname)]
    
    # label
    ret=str(label)
    
    # get hash from filename; remove filename extension
    # assume filename starting with hash
    if filename.index('.')>0:
        hash_fname=filename[0:filename.index('.')] 
    else:
        hash_fname=filename
    ret=ret+delimiter+str(hash_fname)
    # date
    if src_ctime_int is None:
        try:
            src_ctime_int=int(os.path.getctime(src_fname))
        except:
            src_ctime_int=None

    if src_ctime_int is None: 
        ret=ret+delimiter
    else:
        ret=ret+delimiter+datetime.datetime.fromtimestamp(src_ctime_int).strftime("%Y-%m-%d %H:%M:%S")

    #logs
    #print "lines t=",type(lines)
    #  
    for aidx, line in enumerate(lines):
        if len(line)==0:
            continue
        ret=ret+delimiter+line.replace('\t',' ')
        #print aidx,"str idx=",idx
        #ret=ret+comma+line[idx:].replace('\r','')
    #print "End process_text_log(). ret= ", ret
    return ret


            
# expect lines in format .*:{one func call log here}
#   return json string with {"label":<lbl>,"mdate":<date>,"md5":<md5>,"logs":[{},...]}
def process_android_jlog(src_dir, filename, label=None, src_ctime_int=None, start_str="{"):    
    src_fname=os.path.join(src_dir,filename)
    #print "src_fname=",src_fname
    lines = [line.rstrip('\n') for line in open(src_fname)]
    ret='{'
    ret=ret+'"label":"'+str(label)+'"'
    
    # get hash from filename; remove filename extension
    # assume filename starting with hash
    if filename.index('.')>0:
        hash_fname=filename[0:filename.index('.')] 
    else:
        hash_fname=filename
    ret=ret+',"md5":"'+str(hash_fname)+'"'

    if src_ctime_int is None:
        try:
            src_ctime_int=int(os.path.getctime(src_fname))
        except:
            src_ctime_int=None

    if src_ctime_int is None: 
        ret=ret+',"mdate":"None"'
    else:
        ret=ret+',"mdate":"'+datetime.datetime.fromtimestamp(src_ctime_int).strftime("%Y-%m-%d %H:%M:%S")+'"'

    ret=ret+',"logs":['
    comma=""
    #print "lines t=",type(lines)
    if not start_str is None:
        # assume start_str ends with the left { of json
        for aidx, line in enumerate(lines):
            if len(line)==0:
                continue
            idx=None
            try:
                idx=line.index(start_str)
            except:
                print "WARNING: invalid input:", line
                continue
            idx=idx+len(start_str)-1
            jo={}
            try:
                jo=json.loads(line[idx:])
            except:
                print "WARNING: invalid JSON :",line[idx:]
                pass
            ret=ret+comma+json.dumps(jo)
            #print aidx,"str idx=",idx
            #ret=ret+comma+line[idx:].replace('\r','')
            comma=","
    ret=ret+']}'
    #print "End process_android_jlog(). ret= ", ret
    return ret

# check modified date and insert or update a local file in a folder to HDFS
def upsert_a_folder(src_dir, hdfs_tgt_dir, filename, debug): 
    src_fname=os.path.join(src_dir,filename)
    tgt_fname=os.path.join(hdfs_tgt_dir,filename)
    # get target file info
    tgt_dict={}
    try:
        lsl=hdfs.lsl(hdfs_tgt_dir)
        for i in lsl:
            try:
                tgt_dict[ os.path.basename(i["name"])]=i["last_mod"]
            except:
                pass
    except:
        pass
    print "hdfs tgt_dict=",tgt_dict
    
    # get source info
    src_fs=glob.glob(src_fname)
    print "src_fs=",src_fs
    for sf in src_fs:
        # get source file info
        try:
            src_ctime_int=int(os.path.getctime(sf))
        except:
            src_ctime_int=None
        print "src_ctime_int=",src_ctime_int

        src_bfname=os.path.basename(sf)
        tgt_fname=os.path.join(hdfs_tgt_dir,src_bfname)
        # put or rm/put
        try:
            if not src_bfname in tgt_dict:
                #insert new one
                if debug=='N':
                    hdfs.put(sf, hdfs_tgt_dir)
                else:
                    print "DEBUG: put ",src_bfname, "to", hdfs_tgt_dir
            elif src_ctime_int> tgt_dict[src_bfname]:
                if debug=='N':
                    hdfs.rmr(tgt_fname)
                    hdfs.put(sf, hdfs_tgt_dir)
                else:
                    print "DEBUG: replace ",tgt_fname, "by", sf
            else:
                print tgt_fname,"has a newer mdate than",sf,":",src_ctime_int
        except:
            e = sys.exc_info()[0]
            print "Error: ", e
      
            
# check modified date and insert or update a local file to HDFS; debug=Y to check hdfs' mdate and echo info
def upsert_a_file(src_dir, hdfs_tgt_dir, filename, debug): 
    src_fname=os.path.join(src_dir,filename)
    tgt_fname=os.path.join(hdfs_tgt_dir,filename)
    # get source file info
    try:
        src_ctime_int=int(os.path.getctime(src_fname))
    except:
        src_ctime_int=None
    print "src_ctime_int=",src_ctime_int
    # get target file info
    try:
        tgt_stat=hdfs.stat(tgt_fname)
        tgt_mtime=tgt_stat.st_mtime
    except:
        tgt_mtime=None
    print "tgt_mtime=",tgt_mtime
    
    # put or rm/put
    try:
        if tgt_mtime is None:
            #insert new one
            if debug=='N':
                hdfs.put(src_fname, hdfs_tgt_dir)
            else:
                print "DEBUG: put ",src_fname, "to", hdfs_tgt_dir
        elif src_ctime_int> tgt_mtime:
            if debug=='N':
                hdfs.rmr(tgt_fname)
                hdfs.put(src_fname, hdfs_tgt_dir)
            else:
                print "DEBUG: replace ",tgt_fname, "by", src_fname
        else:
            print tgt_fname,"has a newer mdate:", tgt_mtime ,"than",src_fname,":",src_ctime_int
    except:
        e = sys.exc_info()[0]
        print "Error({0}): {1}".format(e.errno, e.strerror)

        

    
    

if __name__ == '__main__':
    __description__ = "utilties for ml"
    main()