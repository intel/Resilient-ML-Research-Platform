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
upload_filename=$1
cid=$2
# this is a base64-ed string
emulator_config=${3}

logfile=$LOG_DIR/${cid}predict.log
src_dir=$UPLOAD_FULL_DIR
tgt_filename=""
fromweb="1"


echo "============================ in emulate.sh " 
# time stampe
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME"
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME" > $logfile 2>&1
STARTTIME=$(date +%s)


ret=-1

echo "INFO: In emulate:" $upload_filename
echo "INFO: In emulate:" $upload_filename >> $logfile 2>&1

tgt_filename=$src_dir/$upload_filename
# generate hash as name
hash_fname=`sha256sum $tgt_filename  | awk -F' ' '{print $1}'`

# generate config file from base64 decoding to NFS
#TBD echo emulator_config=$emulator_config
if [ ! -z "$emulator_config" ] 
then
    echo "Invoke: echo 'emulator_config' | base64 -d > $APK_SANDBOX_IN_DIR/$cid-$hash_fname.apk.config"
    echo "Invoke: echo 'emulator_config' | base64 -d > $APK_SANDBOX_IN_DIR/$cid-$hash_fname.apk.config" >> $logfile 2>&1
    echo "$emulator_config" | base64 -d > $APK_SANDBOX_IN_DIR/$cid-$hash_fname.apk.config
fi

# move apk to NFS
echo Invoke: mv $tgt_filename $APK_SANDBOX_IN_DIR/$cid-$hash_fname.apk
echo Invoke: mv $tgt_filename $APK_SANDBOX_IN_DIR/$cid-$hash_fname.apk >> $logfile 2>&1
mv $tgt_filename $APK_SANDBOX_IN_DIR/$cid-$hash_fname.apk
ret=$?

# if error echo message
if [ $ret -ne 0 ]
then
    echo "Error!! setup apk failed!! ret=${ret}"
fi


# time stampe
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME"
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME" >> $logfile 2>&1
./elapse_time.sh $STARTTIME $logfile

exit $ret

