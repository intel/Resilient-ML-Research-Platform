#!/bin/bash
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
# load config file
sed '/=/!d;/\[\.*/d;/^ *#/d;' < app.config > /tmp/$$.tmp
source /tmp/$$.tmp
rm -rf /tmp/$$.tmp

# cd to working dir
cd $WORKING_DIR
#WORKING_DIR=atdml/tasks

# load params
rid=$1
filename=$2
uploadtype=$3

from_file=$FEATURE_SRC_DIR/$filename
to_hdfs_dir=$FEATURE_DES_DIR/${rid}
logfile=$LOG_DIR/${rid}feature.log
tgt_filename="libsvm_data"
#FTYPE_NGRAM_GZ="N-gram pattern gz"

# will be drop later, use /tmp
OUT_DIR=/tmp/${rid}feature
rm -rf $OUT_DIR
mkdir $OUT_DIR

# time stampe
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME"
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME $uploadtype" > $logfile 2>&1
STARTTIME=$(date +%s)

# ? export for hdfs
export JAVA_HOME=$JAVA_HOME
export HADOOP_HOME=$HADOOP_HOME
export HADOOP_USER_NAME=$HADOOP_USER_NAME


# from ngram gz file
if [ "$uploadtype" == "$FTYPE_NGRAM_GZ" ] || [[ "$uploadtype" == *"Custom"* ]]
then
    #source
    OUT_DIR=$UPLOAD_FULL_DIR/$filename
    #target
    to_hdfs_dir=$RETRIEVE_DATA_DIR/$filename
    # create hdfs folder
    echo INFO: create hdfs folder
    echo INFO: hdfs dfs -mkdir $to_hdfs_dir 
    echo INFO: hdfs dfs -mkdir $to_hdfs_dir >> $logfile 2>&1
    hdfs dfs -mkdir $to_hdfs_dir >> $logfile 2>&1
    # set filename
    to_hdfs_dir=$to_hdfs_dir/$filename
    logfile=$LOG_DIR/${rid}retrieve.log

else
    echo INFO: $from_file
    # for libsvm format file
    # check if zipped
    z_chk=`file $from_file | grep " Zip archive data" | wc -l`
    gz_chk=`file $from_file | grep " gzip compressed data" | wc -l`
    # if 1; ungzip it
    if [ "$z_chk" -eq "1" ];
    then
        echo INFO: unzip $from_file -j $OUT_DIR/
        echo INFO: unzip $from_file -j $OUT_DIR/  >> $logfile 2>&1
        unzip -j $from_file -d $OUT_DIR/ >> $logfile 2>&1
        # get filename
        fname=`ls $OUT_DIR/`
        #chg filename to libsvm_data 
        mv $OUT_DIR/$fname $OUT_DIR/$tgt_filename
    elif [ "$gz_chk" -eq "1" ];
    then
        echo INFO: gzip -c -d $from_file > $OUT_DIR/tmp.txt
        echo INFO: gzip -c -d $from_file > $OUT_DIR/tmp.txt  >> $logfile 2>&1
        gzip -c -d $from_file > $OUT_DIR/tmp.txt 
        rm $from_file
        # get filename
        fname=`ls $OUT_DIR/`
        #chg filename to libsvm_data 
        mv $OUT_DIR/$fname $OUT_DIR/$tgt_filename
    
    else
        mv $from_file $OUT_DIR/$tgt_filename
    fi
    # rename to libsvm_data
     
    # create metadata
    echo INFO: python ml/ml_util.py -md "save_libsvm_metadata" -fn $OUT_DIR/$tgt_filename -r $rid -ky "dic_name_label" #-ip "" -p "" -dn "" -t "" -un "" -pw ""
    echo INFO: python ml/ml_util.py -md "save_libsvm_metadata" -fn $OUT_DIR/$tgt_filename -r $rid -ky "dic_name_label" >> $logfile 2>&1
    python ml/ml_util.py -md "save_libsvm_metadata" -fn $OUT_DIR/$tgt_filename -r $rid -ky "dic_name_label" >> $logfile 2>&1
fi


#upload to HDFS
echo INFO: cleanup hdfs folder
echo hdfs dfs -rm -r $to_hdfs_dir 
#echo hdfs dfs -rm -r $to_hdfs_dir >> $logfile 2>&1
hdfs dfs -rm -r $to_hdfs_dir >> $logfile 2>&1

echo INFO: Upload to HDFS 
echo hdfs dfs -put $OUT_DIR $to_hdfs_dir 
#echo hdfs dfs -put $OUT_DIR $to_hdfs_dir >> $logfile 2>&1
hdfs dfs -put $OUT_DIR $to_hdfs_dir >> $logfile 2>&1
ret=$?

echo "INFO: Upload to HDFS, $to_hdfs_dir/$filename, done!! ret=${ret}"

#remove folder out
#rm -rf $OUT_DIR
# time stampe
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME"
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME" >> $logfile 2>&1
./elapse_time.sh $STARTTIME $logfile

exit 0

