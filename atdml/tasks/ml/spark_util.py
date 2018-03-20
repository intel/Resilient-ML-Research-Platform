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
import json, pickle, gzip

from argparse import ArgumentParser
from time import time
from sklearn.metrics import roc_curve, auc

####pyspark#####
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.util import MLUtils

###pydoop###
import pydoop.hdfs as hdfs


#####import for django database####
#sys.path.append('./db')
#import exec_sqlite

####import our own library####
sys.path.append('./db')
import query_mongo
import ml_util
from ml_util import *

CONF_FILE='../../app.config' # at the base dir of the web
config=ConfigParser.ConfigParser()
config.read(CONF_FILE)
libsvm_filename = config.get("machine_learning","libsvm_alldata_filename")
dnn_filename = config.get("machine_learning","dnn_alldata_filename")



def main():
    
    parser = ArgumentParser(description=__description__)
    parser.add_argument("-sd", "--src_dir", type=str, metavar="source hdfs folder", help="folder contains data", required=False)
    parser.add_argument("-fl", "--fname_list", type=str, metavar="data filename list with wildcard", help="data filename list with wildcard", required=False)
    parser.add_argument("-od", "--out_dir", type=str, metavar="output hdfs folder", help="folder for output", required=False)
    parser.add_argument("-ed", "--exclude_dir", type=str, metavar="hdfs folder for exclusion list", help="hdfs folder for exclusion list", required=False)
    parser.add_argument("-ef", "--exclude_fname_list", type=str, metavar="hdfs filename list for exclusion", help="hdfs filename list for exclusion", required=False)

    parser.add_argument('-sp','--sp_master', type=str, dest='sp_master', help='spark.master'
                , default =config.get('spark', 'spark_master'))
    parser.add_argument('-em','--exe_memory', type=str, dest='exe_memory', help='spark.executor.memory'
                , default =config.get('spark', 'spark_executor_memory'))
    parser.add_argument('-cm','--core_max', type=str, dest='core_max', help='spark.cores.max'
                , default =config.get('spark', 'spark_cores_max'))
    
    args = parser.parse_args()
    
    if args.src_dir:
        hdfs_src_dir = args.src_dir
    else:
        hdfs_src_dir  = None
    if args.fname_list:
        fname_list = args.fname_list
    else:
        fname_list  = '*'
    if args.out_dir:
        hdfs_out_dir = args.out_dir
    else:
        hdfs_out_dir  = '.'
    if args.exclude_dir:
        hdfs_excl_dir = args.exclude_dir
    else:
        hdfs_excl_dir  = None
    if args.exclude_fname_list:
        hdfs_excl_fname_list = args.exclude_fname_list
    else:
        hdfs_excl_fname_list  = '*'

 
    
    return exclude_sample(hdfs_src_dir, fname_list, hdfs_out_dir, hdfs_excl_dir, hdfs_excl_fname_list
    , args.sp_master,config.get('spark', 'spark_rdd_compress'),config.get('spark', 'spark_driver_maxResultSize')
    , args.exe_memory, args.core_max
    , 'Pross Dataset' )
'''
/home/hadoop/spark_latest/bin/spark-submit ml/spark_util.py -sd /user/hadoop/upload/data_retrieved/mc -od /user/hadoop/upload/data_retrieved/vt 
    -ed /user/hadoop/upload/data_retrieved/mc_only_list -ef '*' 
    -fl '17sg_*.gz,adwoleaker_*.gz,adwooa_*.gz,adwoob_*.gz,adwooc_*.gz,agentaj_*.gz,agentam_*.gz,agentbl_*.gz,agenteb_*.gz,agenti_*.gz,agentj_*.gz,agentp_*.gz,agentw_*.gz,agentx_*.gz,airpush_*.gz,androrat_*.gz,anservera_*.gz,anserverc_*.gz,antigate_*.gz,badnumber_*.gz,boqx_*.gz,boxer_*.gz,carej_*.gz,clean_*.gz,deaifraud_*.gz,drddreamlite_*.gz,droidkungfue_*.gz,droidkungfut_*.gz,droidkungfuv_*.gz,fakeinstallerab_*.gz,fakeinstallerad_*.gz,fakeinstallerai_*.gz,fakeinstallerak_*.gz,fakeinstallerap_*.gz,fakeinstalleraq_*.gz,fakeinstallerar_*.gz,fakeinstallerat_*.gz,fakeinstallerau_*.gz,fakeinstallerba_*.gz,fakeinstallerbt_*.gz,fakeinstallercq_*.gz,fakeinstallercu_*.gz,fakeinstallercv_*.gz,fakeinstallerdb_*.gz,fakeinstallerdd_*.gz,fakeinstallerdf_*.gz,fakeinstallerec_*.gz,fakeinstallero_*.gz,fakeinstallerp_*.gz,fakeinstallershtm_*.gz,fakeinstallery_*.gz,fakeinstallerz_*.gz,fakenotify_*.gz,faketokena_*.gz,faketokenc_*.gz,faketokene_*.gz,fakeupdatesa_*.gz,fictusa_*.gz,fictusb_*.gz,fobusa_*.gz,fobusc_*.gz,fobusi_*.gz,fusob_*.gz,ginmasterdropper_*.gz,ginmasterp_*.gz,gluper_*.gz,gsd_*.gz,gsef_*.gz,gumen_*.gz,htmlapp_*.gz,hummingbxx_*.gz,icono_*.gz,kuguo_*.gz,leadbolt_*.gz,lovetheater_*.gz,meds_*.gz,minimob_*.gz,mobidash_*.gz,msega_*.gz,nandrobox_*.gz,plankton_*.gz,rusms_*.gz,sandrorat_*.gz,shedung_*.gz,sheduni_*.gz,skymobia_*.gz,skymobic_*.gz,slockerd_*.gz,slockerk_*.gz,slockerl_*.gz,smforwak_*.gz,smforwaq_*.gz,smforwbo_*.gz,smforwby_*.gz,smsagentd_*.gz,smsagentg_*.gz,smsendaj_*.gz,smsendaq_*.gz,smsendcc_*.gz,smsendcg_*.gz,smsenddh_*.gz,smsenddi_*.gz,smsendeo_*.gz,smsendz_*.gz,smspayo_*.gz,smsregaa_*.gz,smsregi_*.gz,smsregr_*.gz,smsregz_*.gz,smsspyu_*.gz,smsthiefa_*.gz,smsthiefac_*.gz,smsthiefah_*.gz,smsthiefbi_*.gz,smsthiefc_*.gz,smsthiefdg_*.gz,smsthieff_*.gz,smsthiefg_*.gz,smsthiefh_*.gz,smsthiefi_*.gz,smsthiefj_*.gz,smsthiefz_*.gz,stealerb_*.gz,stealerc_*.gz,stealerg_*.gz,taiku_*.gz,utchi_*.gz,vidro_*.gz,vietsmsc_*.gz,wobooo_*.gz,woobooleaker_*.gz,youmia_*.gz' 
    > filter.log

/home/hadoop/spark_latest/bin/spark-submit ml/spark_util.py -sd /user/hadoop/upload/data_retrieved/mc_dyn -od /user/hadoop/upload/data_retrieved/vt_dyn 
    -ed /user/hadoop/upload/data_retrieved/mc_only_list -ef '*' 
    -fl '17sg_*.gz,adwoleaker_*.gz,adwooa_*.gz,adwoob_*.gz,adwooc_*.gz,agentaj_*.gz,agentam_*.gz,agentbl_*.gz,agenteb_*.gz,agenti_*.gz,agentj_*.gz,agentp_*.gz,agentw_*.gz,agentx_*.gz,airpush_*.gz,androrat_*.gz,anservera_*.gz,anserverc_*.gz,antigate_*.gz,badnumber_*.gz,boqx_*.gz,boxer_*.gz,carej_*.gz,clean_*.gz,deaifraud_*.gz,drddreamlite_*.gz,droidkungfue_*.gz,droidkungfut_*.gz,droidkungfuv_*.gz,fakeinstallerab_*.gz,fakeinstallerad_*.gz,fakeinstallerai_*.gz,fakeinstallerak_*.gz,fakeinstallerap_*.gz,fakeinstalleraq_*.gz,fakeinstallerar_*.gz,fakeinstallerat_*.gz,fakeinstallerau_*.gz,fakeinstallerba_*.gz,fakeinstallerbt_*.gz,fakeinstallercq_*.gz,fakeinstallercu_*.gz,fakeinstallercv_*.gz,fakeinstallerdb_*.gz,fakeinstallerdd_*.gz,fakeinstallerdf_*.gz,fakeinstallerec_*.gz,fakeinstallero_*.gz,fakeinstallerp_*.gz,fakeinstallershtm_*.gz,fakeinstallery_*.gz,fakeinstallerz_*.gz,fakenotify_*.gz,faketokena_*.gz,faketokenc_*.gz,faketokene_*.gz,fakeupdatesa_*.gz,fictusa_*.gz,fictusb_*.gz,fobusa_*.gz,fobusc_*.gz,fobusi_*.gz,fusob_*.gz,ginmasterdropper_*.gz,ginmasterp_*.gz,gluper_*.gz,gsd_*.gz,gsef_*.gz,gumen_*.gz,htmlapp_*.gz,hummingbxx_*.gz,icono_*.gz,kuguo_*.gz,leadbolt_*.gz,lovetheater_*.gz,meds_*.gz,minimob_*.gz,mobidash_*.gz,msega_*.gz,nandrobox_*.gz,plankton_*.gz,rusms_*.gz,sandrorat_*.gz,shedung_*.gz,sheduni_*.gz,skymobia_*.gz,skymobic_*.gz,slockerd_*.gz,slockerk_*.gz,slockerl_*.gz,smforwak_*.gz,smforwaq_*.gz,smforwbo_*.gz,smforwby_*.gz,smsagentd_*.gz,smsagentg_*.gz,smsendaj_*.gz,smsendaq_*.gz,smsendcc_*.gz,smsendcg_*.gz,smsenddh_*.gz,smsenddi_*.gz,smsendeo_*.gz,smsendz_*.gz,smspayo_*.gz,smsregaa_*.gz,smsregi_*.gz,smsregr_*.gz,smsregz_*.gz,smsspyu_*.gz,smsthiefa_*.gz,smsthiefac_*.gz,smsthiefah_*.gz,smsthiefbi_*.gz,smsthiefc_*.gz,smsthiefdg_*.gz,smsthieff_*.gz,smsthiefg_*.gz,smsthiefh_*.gz,smsthiefi_*.gz,smsthiefj_*.gz,smsthiefz_*.gz,stealerb_*.gz,stealerc_*.gz,stealerg_*.gz,taiku_*.gz,utchi_*.gz,vidro_*.gz,vietsmsc_*.gz,wobooo_*.gz,woobooleaker_*.gz,youmia_*.gz' 
    > filter.log
    
/home/hadoop/spark_latest/bin/spark-submit ml/spark_util.py -sd /user/hadoop/upload/data_retrieved/mc_sts -od /user/hadoop/upload/data_retrieved/vt_sts 
    -ed /user/hadoop/upload/data_retrieved/mc_only_list -ef '*' -fl '_1st_part_'
    
/home/hadoop/spark_latest/bin/spark-submit ml/spark_util.py -sd /user/hadoop/upload/data_retrieved/mc_dyn_170517 -od /user/hadoop/upload/data_retrieved/vt_dyn_170517 
    -ed /user/hadoop/upload/data_retrieved/mc_only_list -ef '*' -fl '_1st_part_'
    
'''    
    
# ======================================================================= exclude_sample () ============ 
# exclude sample by hasd/key list
def exclude_sample(hdfs_src_dir,fname_list, hdfs_out_dir, hdfs_excl_dir, hdfs_excl_fname_list
    , sp_master, spark_rdd_compress, spark_driver_maxResultSize, sp_exe_memory, sp_core_max
    , jobname , delimitor="_"
    ): 
 
    # clean up output hdfs folder
            
    # get_spark_context
    sc=ml_util.ml_get_spark_context(sp_master
        , spark_rdd_compress
        , spark_driver_maxResultSize
        , sp_exe_memory
        , sp_core_max
        , jobname
        ) 

    t0 = time()
    
    # get exclusion data
    ex_files=os.path.join(hdfs_excl_dir,hdfs_excl_fname_list)
    if ',' in hdfs_excl_fname_list:
        ex_files=""
        comma=""
        for fn in hdfs_excl_fname_list.split(','):
            ex_files=ex_files+comma+os.path.join(hdfs_excl_dir,fn)
            comma=","
    print "INFO: mc only files=",ex_files
    ex_rdd=sc.textFile(ex_files).map(lambda x: (x,x)).distinct().cache()

    # get source files by pattern: the 1st part of filename 
    if fname_list == "_1st_part_":
        f_list=hdfs.ls(hdfs_src_dir)
        f_dict={}
        for i in f_list:
            bname=os.path.basename(i)
            pattern=bname
            if bname.index('_')>0:
                pattern=bname[:bname.index('_')+1]+'*.gz'
            dname=os.path.dirname(i)
            pname=os.path.join(dname,pattern)
            #
            if not pname in f_dict:
                #print "p:",pname
                f_dict[pname]=1
        # create list string
        fname_list=','.join(f_dict)
        
    # for compression
    codec = "org.apache.hadoop.io.compress.GzipCodec"
    # for each source files, exclude data and save to output
    for fn in sorted(fname_list.split(',')):
        if len(fn) ==0:
            continue
        ofn=None
        if not 'hdfs://' in fn:
            src_file=os.path.join(hdfs_src_dir,fn)
            # assume output name is the 1st part of input fname
            ofn=fn.partition(delimitor)[0]
            out_file=os.path.join(hdfs_out_dir, ofn)
        else: # list from "_1st_part_"
            src_file=fn
            bname=os.path.basename(fn)
            ofn=bname.partition(delimitor)[0]
            out_file=os.path.join(hdfs_out_dir, ofn)
         

        print "INFO: out files=",out_file,"ofn=",ofn

        # clean up existing file
        ml_util.ml_clean_up_hdfs_file(out_file)
        
        #load data & set key; may have .<familyname>; left join, filter non-matching; remove key & save
        src_rdd=None
        cnts=0
        label_list=None
        try:
            src_rdd=sc.textFile(src_file) 
            #label_before=src_rdd.map(lambda x: x[:7]).distinct().collect()
            #print "INFO: label_before=",label_before
            src_rdd=src_rdd.filter(lambda x: not x is None) \
                .map(lambda x: (x.split('\t'),x)) \
                .filter(lambda x: len(x[0])>1) \
                .cache()
            #label_list=src_rdd.map(lambda x:x[0][0]).distinct().collect()
            #print "INFO: label_list=",label_list
            src_rdd=src_rdd.map(lambda x: (x[0][1].split('.')[0],x[1])).cache()
            
            #    .map(lambda x: (x.split('\t')[1].split('.')[0],x)).cache()
            cnts=src_rdd.count()
        except:
            print "WARNING:", sys.exc_info()[0]  
        
        #print "INFO: input ["+ofn+"] count=",cnt
        if cnts > 0:    
            f_rdd=src_rdd \
                .leftOuterJoin(ex_rdd) \
                .filter(lambda x:x[1][1] is None) \
                .map(lambda x:x[1][0]).cache()
            cntf=f_rdd.count()
            print "INFO: ["+ofn+"] remain ratio=",cntf,"/",cnts,"=", str(cntf*1.0/cnts)
            if cntf > 0:
                f_rdd.saveAsTextFile(out_file,codec)
            else:
                print "WARNING: filter excluded all data for ["+ofn+"]"
        else:
            print "WARNING: ["+ofn+"] has no data"
    #End for loop
    
    t1 = time()
    print 'INFO: running time: %f' %(t1-t0)
    print 'INFO: Finished!'
    return 0

    
if __name__ == '__main__':
    __description__ = "Processing dataset exclusion"
    main()
