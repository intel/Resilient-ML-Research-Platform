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
uploadtype=$2
out_filename=$3
pattern=$4
hash_list=$5
if [ -z "$6" ]; then
    host=$ip_address
else
    host=$6
fi
if [ -z "$7" ]; then
    port=$port
else
    port=$7
fi
if [ -z "$8" ]; then
    db=$db_name
else
    db=$8
fi
if [ -z "$9" ]; then
    tbl=$tb_name
else
    tbl=$9
fi
usr=${10}
pwd=${11}
model_filename=${12}
keep_flag=${13}
feat_threshold=${14}
#n-gram
#n_gram=$3
# param string
#opt_str=$4
# spark mllib or scikit
#ml_lib=$5

#jstr_proj=${11}

out_dir=$TRAIN_DES_DIR/$rid
_now=$(date +"%Y%m%d_%H%M%S.%N")
logfile=$LOG_DIR/${rid}predict_massive_${_now}.log

#echo "logfile at="$logfile

if [ -z "$model_filename" ]; then
    model_filename=""
else
    model_filename=$TRAIN_DES_DIR/${rid}/${rid}_model/$model_filename
fi


echo "============================ in predict_massive.sh " 
# time stampe
date +"%m/%d/%Y %H:%M:%S $HOSTNAME"
date +"%m/%d/%Y %H:%M:%S $HOSTNAME" > $logfile 2>&1
STARTTIME=$(date +%s)
echo "uploadtype="$uploadtype "
echo "uploadtype="$uploadtype " >> $logfile 2>&1

if [[ $uploadtype == *"in-dynamic"* ]]
then
    jstr_proj=$c_dynamic_proj
    predict_script=$PREDICT_MASSIVE_IN
elif [[ $uploadtype == *"in-static"* ]]
then
    jstr_proj=$c_static_proj
    predict_script=$PREDICT_MASSIVE_IN
fi
ret=-1
# support IN only now

    echo "In massive predict for IN" >> $logfile 2>&1

    # invoke massive retrieve and predict 
    echo python $predict_script  -r $rid -o "$out_dir" -ofn "$out_filename" -hl "$hash_list" -ip "$host" -p $port -dn $db -t $tbl -jp $jstr_proj -un "$usr" -pw "$pwd" -mfn "$model_filename" -ft $feat_threshold
    echo python $predict_script  -r $rid -o "$out_dir" -ofn "$out_filename" -hl "$hash_list" -ip "$host" -p $port -dn $db -t $tbl -jp $jstr_proj -un "$usr" -pw "$pwd" -mfn "$model_filename" -ft $feat_threshold >> $logfile 2>&1
    python $predict_script  -r $rid -o "$out_dir" -ofn "$out_filename" -hl "$hash_list" -ip "$host" -p $port -dn $db -t $tbl -jp $jstr_proj -un "$usr" -pw "$pwd" -mfn "$model_filename" -ft $feat_threshold >> $logfile 2>&1
    ret=$?
    if [ $ret -eq 1 ]
    then
        echo "Error!! $0 failed from model file! Please read $logfile for details. ret=${ret}"
        exit 101
    elif [ $ret -ne 0 ]; then
        echo "ERROR!! $predict_script failed. Please check log for details ret=${ret}" 
        exit $ret
    fi



# check return code and echo message
if [ $ret -ne 0 ]
then
    echo "Error!! $0 failed!! ret=${ret}"
    exit $ret
else
    echo "$0 done! ret=${ret}"
fi


#upload to HDFS
#echo Upload to HDFS 
#hdfs dfs -put out $to_hdfs_dir/$filename
#ret=$?

#echo " *Upload to HDFS, $to_hdfs_dir/$filename, done!! ret=${ret}"

#remove folder out
#rm -rf out

# time stampe
date +"%m/%d/%Y %H:%M:%S $HOSTNAME"
date +"%m/%d/%Y %H:%M:%S $HOSTNAME" >> $logfile 2>&1
./elapse_time.sh $STARTTIME $logfile

# clean up log file
if [ "$keep_flag" == "0" ]
then
    rm -f $logfile
fi

exit 0

