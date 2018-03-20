#!/bin/bash
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
# echo elapsed time based on $1 and current time

STARTTIME=$1 # output of $(date +%s)
LOGFILE=$2 # logfile to echo to

ENDTIME=$(date +%s)
ESEC=$(($ENDTIME - $STARTTIME))
EMIN=$(($ESEC / 60))

#echo $ENDTIME
#echo $ESEC
#echo $EMIN
# change echo to min to min after 31min
if [ $EMIN -gt 30 ]; then
    TU=$EMIN" min"
else
    TU=$ESEC" sec"
fi
echo "Elapsed time $TU"
if [[ !  -z  $LOGFILE  ]]; then
    echo  "INFO: Elapsed time $TU" >> $LOGFILE
fi

