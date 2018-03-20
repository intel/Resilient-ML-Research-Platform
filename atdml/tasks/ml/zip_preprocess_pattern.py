'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''

#####data preprocessing######
from argparse import ArgumentParser
import re, gzip,json, os
from time import time

# Input: 
# line- json string for sample
#   {"label":"<meta-data1>","md5":"<...>","mdate":"<...>","logs":[{"class":"<class1>","method":"<method1>",...},{},... ] }
# Return:
#   an array [<meta-data1>,<meta-data2>,...,["<class1>,<method1>","<class2>,<method2>",...] ]
def preprocess_json(line, meta_list=['label','md5','mdate'], data_key='logs', data_field_list=['class','method'], label_arr=None):
    json_obj=json.loads(line)
    ret_arr=[]
    # for meta
    for k in meta_list:
        if k in json_obj:
            ret_arr.append(json_obj[k])
        else:
            print "ERROR: key "+k+" not found"
            return []
    #for data json array
    if not data_key in json_obj:
        print "ERROR: key "+data_key+" not found"
        return []
    
    # data json arr
    data_j_arr=json_obj[data_key]

    str_arr=[]
    # for each event json object, get data by data_field_list
    for jevt in data_j_arr:
        comma=""
        ret_data=""
        #jevt=json.loads(jevt)
        #print "========= jevt t =",type(jevt),jevt
        # get class,method
        for i in data_field_list:
            if i in jevt:
                #print "key=",i
                ret_data=ret_data+comma+jevt[i]
                comma=","
        str_arr.append(ret_data)
    ret_arr.append(str_arr)

    return ret_arr
    
# Input: 
# line- data format for each line:
#   <metadata1=label>\t<metadata2=md5>\t<metadata3=date>\t<metadata=filetype>
#       ...\t<metadataN>\t<line1>\t...\t<lineM>\n
#   each line for one sample (delimitor='\n').
#       log lines for one sample is delimited by delimitor='\t'
#   metadata_count-: the count of meta-data, they should be before the log data
#   pattern_str- regular expression match, matching groups will be a feature
#   delimitor- delimiter for lines (log)
#   label_idx- index/position for label in line
#   label_arr- python arry to verify data (e.g. ['clean','dirty'])
#   convert2dirty- converting non-clean label to "dirty")
# Return:
#   an array [meta-data1,meta-data2,...,str_arr]
def preprocess_pattern(line, metadata_count, pattern_str, delimitor='\t', label_idx=0
        , label_arr=None, convert2dirty=None):
    # get token between "I/AndroidATD(pid): [API]" and "["
    #pattern_str=r'^I/AndroidATD\([ ]*[\d]+\): \[API\] (.*) \[.*'
    #t0 = time()
    # escape for \t
    delimitor=delimitor.decode('string-escape')
    
    if not line:
        return []
    
    str_lines=line.split(delimitor)
    
    #print "pattern_str=",pattern_str+"<=="
    #print "str_lines=",len(str_lines)
    #print "metadata_count=",metadata_count
    #print "delimitor=",delimitor+"<=="
    #print "label_idx=",label_idx
    #print "label_arr=",label_arr
    
    metadata_count=int(metadata_count)
    
    # check if metadata_count > numb of splits
    if len(str_lines)< metadata_count:
        return []
        
        
    #filter by label, if label_arr provided

    elif label_idx>=0 and label_arr and not str_lines[label_idx] in label_arr: 
        # converting non-clean label to dirty
        if convert2dirty=="Y" :
            if len(label_arr)==2:
                str_lines[label_idx]="dirty"
            else:
                return []
    #print "label_idx=",label_idx
    #print "label_arr=",label_arr        
    ret=str_lines[:metadata_count]
    
    # fillout metadata (label, md5, date, filetype etc)
    #for i in range(0,metadata_count):
    #    ret.append(str_lines[i])
        
    if pattern_str == '(.*)': # bypass pattern
        #print 'bypass pattern ................., data len=', len(str_lines[metadata_count:])
        ret.append(str_lines[metadata_count:])

        #t1 = time()
        #print 'INFO: time for preprocess: %f' %(t1-t0)
        return ret
    else:
        pattern=re.compile(pattern_str, re.IGNORECASE)
        
        
    # for ngram data ===========    
    str_arr=[]
    # match pattern and save to arr, discard non-matches
    for token in str_lines[metadata_count:]:
        m=pattern.match(token)
        if m:
            grps=m.groups()
            #print "grps len=",len(grps), "token=", token
            #print "grps=",grps
            if grps and len(grps)>0:
                ss=""
                comma=""
                for s in grps: # concatenate all matching subgroups
                    if s and len(s)>0:
                        ss = ss+comma+s
                        comma=","
                if len(ss)>0:
                    str_arr.append(ss)
                #str_arr.append(m.group(0)) # whole matching string
                #str_arr.append(m.group(1))

    ret.append(str_arr)
    return ret

def convert_to_line(file_handle, metadata_count, delimitor='\t', replace_char=" "):
    # escape for \t
    delimitor=delimitor.decode('string-escape')
    cnt=0
    out_line=""
    # chk if one line only
    for aline in file_handle:
        if len(aline.strip())>1:
            if cnt > 1:
                break 
            out_line=aline # for one line return
            cnt=cnt+1

    if cnt>1:
        out_line=""
        #print "INFO: line count=",cnt
        file_handle.seek(0)
        # add meta data
        for i in range(1,metadata_count):
            out_line=out_line+delimitor
        # conversion here
        for aline in file_handle:
            aline=aline.strip().replace(delimitor,replace_char)
            out_line=out_line+delimitor+aline
        #print "INFO: len(out_line)=",len(out_line)
    return out_line

# convert text file to one line
def convert_to_line_by_bash(gzfilename,metadata_count,ln_delimitor):
    to_one_line=" | sed -e 's/[[:space:]]*$//' | sed 's/\\r/ /g' | sed 's/\\t/ /g' | tr '\\n' '\\t' | sed 's/\\t$/\\n/'"
    # remove tailing \n
    ret=os.popen("zcat '"+gzfilename+"' "+to_one_line).read().strip()

    # add meta data
    for i in range(0,metadata_count):
        ret=ln_delimitor+ret
    #print "out_line=",ret.replace('\t',',')
    return ret
    
# test only
def main():
    # parse arguments
    parser = ArgumentParser(description=__description__)
    parser.add_argument('-tf', '--tf', type = str, metavar = 'test file', help = 'test file', required =False)
    parser.add_argument("-lba", "--label_arr", type=str, metavar="string array for label", help="string array for label; to verify data too"
        , default ="['clean','dirty']", required=False)
    parser.add_argument("-ptn", "--pattern_str", type=str, metavar="regular express pattern to extract string"
        , help="regular express pattern to extract string"
        , default =r'^I/AndroidATD\([ ]*([\d]+)\): \[API\] (.*) \[(.*)', required=False)
    
    args = parser.parse_args()
    
    print "args.tf=",args.tf
    # open file
    lines = [line.rstrip('\n') for line in gzip.open(args.tf)]
    for l in lines:
        ret= preprocess_json(json.loads(l))
        print "ret=",ret

    '''
    # open file
    f=open(args.tf, 'r')
    ret=convert_to_line(f,3)
    print "len(ret)=",len(ret)
    f.close()  
    
    for l in ret.split("\t"):
        print "l=",l
    
    '''
    
    '''
    # check lines
    cnt=0
    for aline in f:
        if len(aline.strip())>1:
            cnt=cnt+1
    print "cnt=",cnt
    out_line=""
    sep_char=""
    if cnt>1:
        f.seek(0)
        for aline in f:
            aline=aline.strip().replace("\t"," ")
            #print "aline=",aline
            out_line=out_line+sep_char+aline
            sep_char="\t"
        print "len(out_line)=",len(out_line)
    '''
    
    '''
    for line in f:
        ret=preprocess_pattern(line, metadata_count=3
            , pattern_str=args.pattern_str
            , label_arr=args.label_arr)
        print "=========================="
        print "ret=",ret,"   ============"
    '''
    
if __name__ == '__main__':
    __description__ = "preprocessing"
    main()    
    