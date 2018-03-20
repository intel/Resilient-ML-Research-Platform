#!/bin/bash
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
# load params
source ./submit_apk.config

#pure APK submission; no ML prediction
act_dir="atdml/api/apk"
df_dir="atdml/api/df"
fname=$1

# check parameter count
if [ $# -lt 1 ]
then
    echo "APK filename is required !"
    exit -1
fi
# check if file exist
if [ ! -f $fname ]
then
    echo "File '"$fname"' not found!"
    exit -1
fi
econfig64=""
# check if $emulater_config defined and check if file exist
if [ ! -z $emulater_config_fname ]
then
    if [ -f $emulater_config_fname ]
    then
        # encode config file
        econfig64=`base64 $emulater_config_fname`
        #echo $econfig64
    fi
fi

# submit APK =======================
#echo 
sub_json=`curl -s -S -X POST http://$web_host/$act_dir/ -u $uid:$passwd -H "Content-Type: multipart/form-data" -F "docfile=@$fname" --form-string "_desc=from script" --form-string "_emulater_config=$econfig64"`
status=`echo $sub_json | grep -Po '"status":.*?[^\\\\]"' | awk '{split($0,a,":");print a[2]}' | tr -d '"'`
# check if submission status
if [ ! "$status" == "submitted" ]
then
    echo "Error occured. Submit_status="$sub_json
    exit -2
#else
#echo Submit_status=$sub_json
fi
# get id =======================
 
cid=`echo $sub_json | grep -Po '"id":.*?[^\\\\]"' | awk '{split($0,a,":");print a[2]}' | tr -d '"'`
# polling apk execution =======================
echo "id="$cid" Polling job status:"
counter=0
# poll status every $sleep_interval seconds =======================
while [ $counter -lt $wait_cycle ]
do
    echo -n .
    # get status
    sts_json=`curl -s -S -X  GET  http://$web_host/$act_dir/$cid/ -u $uid:$passwd`
    #echo status_return=$sts_json
    #echo echo $sts_json | grep -Po '"status":.*?[^\\]"' | awk '{split($0,a,":");print a[2]}' | tr -d '"'
    status=`echo $sts_json | grep -Po '"status":.*?[^\\\\]"' | awk '{split($0,a,":");print a[2]}' | tr -d '"'`
    counter=$[$counter+1]
    if [[ "$status" != "submitted" ]]
    then
        echo Status=$status
        counter=$wait_cycle
    else
        sleep $sleep_interval
    fi
done

# retrieve execution log & zip files
if [[ "$status" == "succeed" ]]
then
    echo "Download execution log to '"$log_output_dir/$cid$log_suffix"'"
    sleep $sleep_interval
    mkdir -p $log_output_dir
    curl -s -S -X GET http://$web_host/$log_dir/$cid/_e_$cid/-1/ -u $uid:$passwd -o $log_output_dir/$cid$log_suffix
    
    # check if flag of zip is available
    has_zip=$(head -n 1 $emulater_config_fname)
    if [[ "$has_zip" == "SAVE_RUNTIME_FILES=1" ]]
    then
        echo "Download output to '"$log_output_dir/$cid"_output.zip'"
        curl -s -S -X GET http://$web_host/$df_dir/$cid/ -u $uid:$passwd -o $log_output_dir/$cid"_output.zip"
    fi
fi
echo ""
