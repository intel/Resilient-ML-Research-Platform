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
out_f=$4
out_prob=$5
out_it=$6
out_comb=$7
ctype=$8
ds_id=$9

spark_cmd=$SPARK_SUBMIT #/home/hadoop/spark-1.2.0-bin-hadoop2.4/bin/spark-submit
hdfs_dir=$HADOOP_MASTER$FEATURE_DES_DIR/${rid}
sparkurl=$SPARK_URL
# if ds_id is set, it is an option
if [[ $ds_id ]]
then
    hdfs_dir=$HADOOP_MASTER$FEATURE_DES_DIR/${ds_id}
fi


logfile=$LOG_DIR/${rid}feature_importance.log

#FEATURE_IMPO_FIRM=feature_importance_FIRM.py
#FEATURE_IMPO_2WAYS=feature_importance_2ways.py
#FEATURE_IMPO_COMB=feature_importance_combine.py

FIRM=$FEATURE_IMPO_FIRM
TWO_WAY=$FEATURE_IMPO_2WAYS

# generic feature importance program
if [[ "$uploadtype" == *"pattern"* ]] ; then
    FIRM=$FEATURE_IMPO_GEN_FIRM
    TWO_WAY=$FEATURE_IMPO_GEN_IP
fi
# time stampe
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME"
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME" > $logfile 2>&1
STARTTIME=$(date +%s)
echo File type: $uploadtype
ret=-1

# create all files
if [ "$ctype" == "" ]; then
    echo Invoke Spark: $spark_cmd $FIRM -f $hdfs_dir -n $filename -o $out_f -r $rid -sp $sparkurl -u "$uploadtype" -dsid "$ds_id" 
    echo Invoke Spark: $spark_cmd $FIRM -f $hdfs_dir -n $filename -o $out_f -r $rid -sp $sparkurl -u "$uploadtype" -dsid "$ds_id" >> $logfile
    $spark_cmd $FIRM -f $hdfs_dir -n $filename -o $out_f -r $rid -sp $sparkurl -u "$uploadtype" -dsid "$ds_id"  >> $logfile 2>&1
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "ERROR: $FIRM failed. Please check log for details ret=${ret}" 
        echo "ERROR: $FIRM failed. Please check log for details ret=${ret}" >> $logfile 2>&1
        exit $ret
    fi
    echo Invoke Spark: $spark_cmd $TWO_WAY -f $hdfs_dir -n $filename -pb $out_prob -it $out_it -r $rid -sp $sparkurl -u "$uploadtype" -dsid "$ds_id" 
    echo Invoke Spark: $spark_cmd $TWO_WAY -f $hdfs_dir -n $filename -pb $out_prob -it $out_it -r $rid -sp $sparkurl -u "$uploadtype" -dsid "$ds_id" >> $logfile
    $spark_cmd $TWO_WAY -f $hdfs_dir -n $filename -pb $out_prob -it $out_it -r $rid -sp $sparkurl -u "$uploadtype" -dsid "$ds_id"  >> $logfile 2>&1
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "ERROR: $TWO_WAY failed. Please check log for details. ret=${ret}" 
        echo "ERROR: $TWO_WAY failed. Please check log for details. ret=${ret}" >> $logfile 2>&1
        exit $ret
    fi
fi

# create combine file & for refresh
echo Invoke: python $FEATURE_IMPO_COMB -f $out_f -it $out_it -pb $out_prob -s $out_comb -r $rid -u "$uploadtype" -dsid "$ds_id"  
echo Invoke: python $FEATURE_IMPO_COMB -f $out_f -it $out_it -pb $out_prob -s $out_comb -r $rid -u "$uploadtype" -dsid "$ds_id"  >> $logfile
python $FEATURE_IMPO_COMB -f $out_f -it $out_it -pb $out_prob -s $out_comb -r $rid -u "$uploadtype" -dsid "$ds_id"   >> $logfile 2>&1
ret=$?


# check return code and echo message
if [ $ret -ne 0 ]
then
    echo "Error: $FEATURE_IMPO_COMB failed. Please check log for details. ret=${ret}" 
    echo "Error: $FEATURE_IMPO_COMB failed. Please check log for details. ret=${ret}" >> $logfile 2>&1
    exit $ret
else
    echo "INFO: Feature importance script done!  ret=${ret}"
    echo "INFO: Feature importance script done!  ret=${ret}" >> $logfile 2>&1
fi

# time stampe
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME"
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME" >> $logfile 2>&1
./elapse_time.sh $STARTTIME $logfile

exit 0

