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
filename_prefix=$2
uploadtype=$3
algtype=$4
opts=$5
ds_id=$6
ml_model=$7

# set src here, need a flag in model to know starting from
hdfs_dir=$HADOOP_MASTER$FEATURE_DES_DIR/${rid}

# if ds_id is set, it is an option
if [[ $ds_id ]]
then
    hdfs_dir=$HADOOP_MASTER$FEATURE_DES_DIR/${ds_id}
fi


out_dir=$TRAIN_DES_DIR/$rid
#from_file=$FEATURE_SRC_DIR/$filename_prefix
logfile=$LOG_DIR/${rid}train.log
train_alg=$TRAIN
hdfs_model_dir=$HADOOP_MASTER$HDFS_MODEL_DIR/$rid
spark_cmd=$SPARK_SUBMIT
sparkurl=$SPARK_URL

#ML_QUEUE_DIR=/home/django/myml/media/ml_queue
#DNN_PREPROCESS=ml/preprocess_dnn.py
logger_file=$out_dir/$rid"_logger.csv"
hdfs_fname=$dnn_alldata_filename

fromweb="0"
if [ $rid -gt 0 ]
then
    fromweb="1"
fi
ret=-1

# time stampe
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME"
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME" > $logfile 2>&1
STARTTIME=$(date +%s)
echo INFO: File type: $uploadtype

# source/target filenames
src_fname_data=$hdfs_dir/$ds_id$dnn_data_suffix
src_fname_label=$hdfs_dir/$ds_id$dnn_label_suffix
src_fname_info=$hdfs_dir/$ds_id$dnn_info_suffix
tgt_fname_data=$ML_QUEUE_DIR/$rid$dnn_data_suffix
tgt_fname_label=$ML_QUEUE_DIR/$rid$dnn_label_suffix
tgt_fname_info=$ML_QUEUE_DIR/$rid$dnn_info_suffix
file_in_local=0

# check if data file exist in hdfs ?
f_count=`hdfs dfs -ls $hdfs_dir/$ds_id'_dnn_*.npy.gz'  | wc -l`
# file not found, then generate data file ?
if [ "$f_count" -eq 0 ]
then
    # generate data files
    echo INFO Invoke Spark: $spark_cmd $DNN_PREPROCESS -f $hdfs_dir -n $hdfs_fname -o $out_dir -r $rid  -sp $sparkurl -pm "$opts"  -dsid "$ds_id"
    echo INFO Invoke Spark: $spark_cmd $DNN_PREPROCESS -f $hdfs_dir -n $hdfs_fname -o $out_dir -r $rid  -sp $sparkurl -pm "$opts" -dsid "$ds_id" >> $logfile
    $spark_cmd $DNN_PREPROCESS -f $hdfs_dir -n $hdfs_fname -o $out_dir -r $rid  -sp $sparkurl -pm "$opts" -dsid "$ds_id" >> $logfile  2>&1
    ret=$?

    # check return code and echo message
    if [ $ret -ne 0 ]
    then
        echo "ERROR: DNN preprossing failed!! ret=${ret}"
        echo "ERROR: DNN preprossing failed!! ret=${ret}" >> $logfile
        exit $ret
    else
        echo "INFO: DNN preprocessing done! ret=${ret}"
        echo "INFO: DNN preprocessing done! ret=${ret}" >> $logfile
        # upload to HDFS
        echo hdfs dfs -put $out_dir/$ds_id$dnn_data_suffix $src_fname_data >> $logfile
        hdfs dfs -put $out_dir/$ds_id$dnn_data_suffix $src_fname_data
        hdfs dfs -put $out_dir/$ds_id$dnn_label_suffix $src_fname_label
        hdfs dfs -put $out_dir/$ds_id$dnn_info_suffix $src_fname_info
        echo "INFO: upload to HDFS done! ret=${ret}"
        echo "INFO: upload to HDFS done! ret=${ret}" >> $logfile
        file_in_local=1
    fi
fi


# clean up files
rm -f $ML_QUEUE_DIR/$rid"_dnn_"*".npy.gz"

if [ "$file_in_local" -eq 1 ] 
then
    # move data file to target server
    echo mv $out_dir/$ds_id$dnn_data_suffix $tgt_fname_data
    echo mv $out_dir/$ds_id$dnn_data_suffix $tgt_fname_data >> $logfile 2>&1
    mv $out_dir/$ds_id$dnn_data_suffix $tgt_fname_data
    mv $out_dir/$ds_id$dnn_label_suffix $tgt_fname_label
    mv $out_dir/$ds_id$dnn_info_suffix $tgt_fname_info

else
    # get files from HDFS
    echo hdfs dfs -get $src_fname_data $tgt_fname_data
    echo hdfs dfs -get $src_fname_data $tgt_fname_data >> $logfile 2>&1
    hdfs dfs -get $src_fname_data  $tgt_fname_data >> $logfile 2>&1
    hdfs dfs -get $src_fname_label $tgt_fname_label >> $logfile 2>&1
    hdfs dfs -get $src_fname_info  $tgt_fname_info >> $logfile 2>&1
    ret=$?
    # check return code and echo message
    if [ $ret -ne 0 ]
    then
        echo "ERROR: data files download failed!! ret=${ret}"
        echo "ERROR: data files download failed!! ret=${ret}" >> $logfile 2>&1
        exit $ret
    else
        echo "INFO: Done file copy! ret=${ret}"
        echo "INFO: Done file copy! ret=${ret}" >> $logfile 2>&1
    fi
fi
# clean up logger file
echo "epoch,acc,loss,val_acc,val_loss" > $logger_file

# create <id>_jog.json file and mv to cuda server: has ml_model, ml_opts inside
#echo "ml_opts="$opts >> $logfile 2>&1
echo "INFO: Create ML Options file" >> $logfile 2>&1
echo "INFO: opts="$opts
echo $opts > $ML_QUEUE_DIR/$rid"_dnn_opts.json"
# model file will trigger the training job
echo "INFO: Create DNN Model file" >> $logfile 2>&1
echo "INFO: model="$ml_model
echo $ml_model > $ML_QUEUE_DIR/$rid"_dnn_model.json"

# service will update status


ret=$?
# check return code and echo message
if [ $ret -ne 0 ]
then
    echo "ERROR: train_dnn failed!! ret=${ret}"
    exit $ret
else
    echo "INFO: Done DNN job submittion! ret=${ret}"
    echo "INFO: Done DNN job submittion" >> $logfile 2>&1
fi


# time stampe
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME"
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME" >> $logfile 2>&1
./elapse_time.sh $STARTTIME $logfile

exit 0

