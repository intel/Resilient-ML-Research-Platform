#!/bin/bash
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
# check filesize here
fname=$1
fsize=`ls -l $fname | awk '{split($0,a," ");print a[5]}'`

if [ $fsize -le 0 ]
then
    echo "Error!! The size of target file '"$fname"' is zero"
    exit -1
fi

gzip $fname
ret=$?
if [ $ret -ne 0 ]
then
    echo "Error!! gzip failed!! ret=${ret}"
    exit $ret
fi
fname=$fname.gz
