'''
/****************************************************************************
* INTEL CONFIDENTIAL
* Copyright 2014-2015 Intel Corporation All Rights Reserved.
*
* The source code contained or described herein and all documents related to
* the source code ("Material") are owned by Intel Corporation or its
* suppliers or licensors.  Title to the Material remains with Intel
* Corporation or its suppliers and licensors.  The Material contains trade
* secrets and proprietary and confidential information of Intel or its
* suppliers and licensors.  The Material is protected by worldwide copyright
* and trade secret laws and treaty provisions.  No part of the Material may
* be used, copied, reproduced, modified, published, uploaded, posted,
* transmitted, distributed, or disclosed in any way without Intel's prior
* express written permission.
*
* No license under any patent, copyright, trade secret or other intellectual
* property right is granted to or conferred upon you by disclosure or
* delivery of the Materials, either expressly, by implication, inducement,
* estoppel or otherwise.  Any license under such intellectual property rights
* must be express and approved by Intel in writing.
****************************************************************************/
'''
#! /usr/bin/python
# standard library imports
from argparse import ArgumentParser
import json

def arg_parser(parser):
    # input output file info
    #parser.add_argument("-f", "--folder", type=str, metavar="folder of features", help="folder contains features, hdfs://xxx.com:9000/user/fea", required=False)
    #dataset info
    #parser.add_argument("-fn", "--fname", type=str, metavar="input filename", help="input filename for data", required=False)
    parser.add_argument("-fp", "--featuring_params", type=str, metavar="parameters for custom featuring"
        , help="parameters for custom featuring", required=False)
    parser.add_argument("-ln", "--line", type=str, metavar="data", help="a line of data", required=False)
    
    return parser.parse_args()
    
def main():  # ============= =============  ============= =============
    # parse arguments
    parser = ArgumentParser(description=__description__)
    args = arg_parser(parser)

    if args.featuring_params:
        featuring_params = args.featuring_params
    else:
        featuring_params  = None
    #if args.featuring_params:
    #    featuring_params = args.featuring_params
    #else:
    #    featuring_params  = None
    if args.line:
        line = args.line
    else:
        line  = None

    
    out=featuring(line, featuring_params)
    print out
    return

# ================================================================================== train () ============ 

# Input: 
#    line string: "<item0>\t<item1>\t<item2>\t<evt type> <entropy numb>\t<evt type> <entropy numb>\t..."
# Return:
#    array: [<item0>,<item1>,<item2>,[<evt type>-<entropy bucket>,<evt type>-<entropy bucket>,...]]
def featuring(line, featuring_params, delimitor='\t', data_idx=3):
    jparams={}
    ret_arr=[]
    log_arr=[]
    type=''
    if featuring_params and len(featuring_params)>0:
        try:
            jparams=json.loads(featuring_params)
        except Exception as e:
                print "ERROR: user module error.", e.__doc__, e.message
                return -200
    if 'type' in jparams:
        type=jparams['type']
    # chk empty 
    if not line or len(line)==0:
        return []

    str_lines=line.split(delimitor)
    
    # meta data
    for i in str_lines[:data_idx]:
        ret_arr.append(i)

    # assume to have 2 or 7 values
    
    if type == 'evt_entropy': # evt+entropy
        # bucket the entropy
        for item in str_lines[data_idx:]:
            val=item.split(' ')
            val_cnt=len(val)
            entropy=0
            evt=""    
            try:
                if val_cnt>=7:
                    evt=val[2]
                    entropy=val[3]
                elif val_cnt==2:
                    evt=val[0]
                    entropy=val[1]
                entropy=float(entropy)
            except:
                pass
            if entropy >=0.9:
                log_arr.append(evt+"_0.9+")
            elif entropy>=0.8:
                log_arr.append(evt+"_0.8+")
            elif entropy>=0.6:
                log_arr.append(evt+"_0.6+")
            elif entropy>=0.4:
                log_arr.append(evt+"_0.4+")
            elif entropy>=0.2:
                log_arr.append(evt+"_0.2+")
            else:
                log_arr.append(evt+"-")
    else: # do nothing
        log_arr=str_lines[data_idx:]
        
    ret_arr.append(log_arr)
    
    return ret_arr

if __name__ == '__main__':
    __description__ = "custom featuring"
    main()    