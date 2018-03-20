'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''
from django.shortcuts import render
from django.template import RequestContext
from django.http import HttpResponseRedirect, HttpResponse
from django.core.urlresolvers import reverse
from django.core import serializers
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Q
import subprocess, datetime
import json, time, os, glob
import _list
from atdml.models import Document
from atdml.forms import DocumentForm
MAX_READ_SIZE=500000 # 500Kb


#============================================================= job_logs ==================
def job_logs(request, rid, perm,disabled4reader, cid=None ):
    print 'in job_logs(), rid=', rid
    document =_list.get_ds_doc(rid, perm)
    print "document..=",document
    if not document:
        return HttpResponseRedirect(reverse('list'))

    filename=document.filename
    
    train_id=None
    if cid is None:
        prd_id=request.POST.get("_prd_id")
    else:
        prd_id=cid
    train_id=document.train_id
    #print "prd_id=",prd_id
 
    #get log files
    dir_str=os.path.join(settings.LOG_FOLDER,rid+"[a-z]*.log")
    
    # get a list of filenames 
    alllist=glob.glob(dir_str)
    pipeline=['retrieve','feature','pca','train','multi_run','feature_importance']
    
    #print file_list
    content1st=""
    file_list=[]
    prdct_lst=[]
    exec_lst=[]
    if len(alllist) > 0:
        # remove path and leading rid
        #file_list=[ os.path.basename(f).replace(rid,'').replace('.log','') for f in sorted(alllist) ]
        ava_list=[ os.path.basename(f).replace(rid,'').replace('.log','') for f in alllist ]

        # filter file list and keep pipeline order
        for i in pipeline:
            for j in ava_list:
                if i==j:
                    file_list.append(i)
        #print file_list
        content1st=getlog(rid,file_list[0],0, perm ,None)
        
    #add predict    
    print "document.file_type=",document.file_type
    prdct_doc_lst=Document.objects.all().filter(file_type__contains="predict", train_id=rid).order_by('-id')[0:200]
    
    for i in prdct_doc_lst:
        print i.id, i.filename
    print "prd_id=",prd_id
    
    if len(prdct_doc_lst)>0:
        #prdct_lst=[ (str(d.id), d.filename) for d in sorted(prdct_doc_lst,reverse=True) ]
        prdct_lst=[ (str(d.id), d.filename) for d in prdct_doc_lst ]
        prdct_lst=sorted(prdct_lst,reverse=True)
        # trick to set latest predict_id for negative predict id
        if prd_id and prd_id.startswith('-') and prd_id[1:].isdigit():
            plist = [i[0] for i in prdct_lst]
            prd_id=plist[0]
            #print "plist=",plist
            #print "prd_id2=",prd_id
        file_list.append("predict")
            #print "prdct_lst=",prdct_lst
        #for ensemble
        if len(content1st)==0:
            content1st=getlog(prdct_lst[0][0],file_list[0],0, perm ,None)

    # find execution log ============== =============
    exec_doc_lst=Document.objects.all().filter(file_type__contains="predict", train_id=rid, desc="has_exe_log").order_by('-id')[0:200]
    if len(exec_doc_lst)>0:
        exec_lst=[ (str(d.id), d.filename) for d in exec_doc_lst ]
        exec_lst=sorted(exec_lst,reverse=True)
        file_list.append("execution log")
    
    jopts=document.ml_opts
    if jopts:
        jopts=json.loads(document.ml_opts)
        if "learning_algorithm" in jopts:
            jopts["learning_algorithm"]=jopts["learning_algorithm"].title().replace("_"," ")
    pca_jopts=document.ml_pca_opts
    if pca_jopts:
        pca_jopts=json.loads(document.ml_pca_opts)

    
    #print 'exec_lst=',exec_lst
    return render(request,
        'atdml/joblogs.html',
        {'document': document 
                ,'file_list':file_list, 'content1st':content1st, 'prdct_lst':prdct_lst, 'exec_lst':exec_lst
                    , 'disabled4reader':disabled4reader, 'perm':perm, 'prd_id':prd_id, 'train_id':train_id
                    ,'jopts':jopts,'pca_jopts':pca_jopts
                    #, 'msg_error':msg_error, 'msg_success': msg_success
        },  
    )

#============================================================= ae_logs ==================
def ae_logs(request, cid, perm,disabled4reader ):
    print 'in ae_logs(), cid=', cid

    exec_lst=[]
    prdct_lst=[]
    # hide predict output for AWS
    file_list=["execution log"]

    # find execution log ============== ============= 
    if perm=="5": # admin get all APK; return last 1000 items only
        exec_doc_lst = Document.objects.all().filter(Q(file_type="emulate") | Q(desc="has_exe_log") ).order_by('-id')[0:1000]
        prdct_doc_lst= Document.objects.all().filter((Q(file_type="emulate") | Q(desc="has_exe_log")), ~Q(train_id=-1)).order_by('-id')[0:200]
    else:
        exec_doc_lst = Document.objects.all().filter((Q(file_type="emulate") | Q(desc="has_exe_log")), acl_list=perm).order_by('-id')[0:1000]
        prdct_doc_lst= Document.objects.all().filter((Q(file_type="emulate") | Q(desc="has_exe_log")), ~Q(train_id=-1), acl_list=perm).order_by('-id')[0:200]

    if len(exec_doc_lst)>0:
        exec_lst=[ (str(d.id), d.filename) for d in exec_doc_lst ]
        exec_lst=sorted(exec_lst,key=lambda x: int(x[0]),reverse=True)
    
    return render(request,
        'atdml/ae_logs.html',
        {   "file_list":file_list, 'exec_lst':exec_lst, "cid":cid, "prdct_lst":prdct_lst
          , 'disabled4reader':disabled4reader, 'perm':perm     
        },  
    )
    
#============================================================= get_log_file text version ==================
def get_log_file(rid, ltype, offset, perm,disabled4reader):
    print "in get_log_file(), rid=", rid,",ltype=",ltype, "offset=", offset
    filename=get_log_fname(rid,ltype)
    fsize,logtxt=get_log_content(filename,ltype,offset)
    #print "logtxt=",logtxt
    return HttpResponse(logtxt, content_type="application/text")
    
#============================================================= get_log_fname ==================
def get_log_fname(rid,ltype):
    filename=""
    # prediction log 
    if ltype.isdigit():
        filename=os.path.join(settings.LOG_FOLDER,''+ltype+'predict.log')
    elif ltype.startswith("_e_"): # get execution log from EXEC_LOG_FOLDED by pattern .only.log
        wfname=os.path.join(settings.EXEC_LOG_FOLDER,ltype[3:],  settings.EXEC_LOG_FNAME) #'*.only.log')
        elist=glob.glob(wfname)
        if len(elist)>=1:   
            filename=elist[0]
        else: # check result folder if not found
            wfname=os.path.join(settings.EXEC_RESULT_FOLDER,ltype[3:],  settings.EXEC_LOG_FNAME)
            elist=glob.glob(wfname)
            if len(elist)>=1:   
                filename=elist[0]
            else: #not found
                filename=wfname
    else:     
        filename=os.path.join(settings.LOG_FOLDER,''+rid+ltype+'.log')
    return filename
    
#============================================================= get_log_content ==================
def get_log_content(filename,ltype,offset):
    logtxt=''
    fsize=0
    offset=int(offset)
    try:
        with open(filename,'r') as fp:
            # get size
            fp.seek(0,2)
            fsize=fp.tell()
            fp.seek(0)

            # only read max to MAX_READ_SIZE
            read_size=fsize
            if fsize > MAX_READ_SIZE:
                read_size=MAX_READ_SIZE
                
            if offset == 0 : # read up to MAX_READ_SIZE
                #for i, line in enumerate(fp.readlines()):
                #    logtxt=logtxt+line
                # for huge file, show only first and last parts.
                if fsize > MAX_READ_SIZE:
                    logtxt=fp.read(MAX_READ_SIZE/2)                       
                    fp.seek(-1*(MAX_READ_SIZE/2),2)
                    logtxt=logtxt+'\n\n\n...\n...\n...\nNOTE: Part of content was skipped due to the size of the log.\n  Total Size = ' \
                        +str(fsize) +' byte. \n  Please click "Download" to view full content locally.\n...\n...\n...\n\n\n'
                    logtxt=logtxt+fp.read()
                else:
                    logtxt=fp.read(read_size)
            elif offset == -1 : # read all for refresh button
                logtxt=fp.read()
            else: # read start from offset
                if offset < fsize:
                    fp.seek(offset)
                    logtxt=fp.read(read_size)
                #endln=i
    except Exception as e:
        print "e=",e
        #logtxt = 'Log file not found or read error! '+ str(e)
        # for AWS:
        logtxt = 'Log file not found! '
    return fsize,logtxt
    
#============================================================= getlog ==================
def getlog(rid,ltype,offset, perm,disabled4reader):
    print "in getlog(), rid=", rid,",ltype=",ltype, "offset=", offset
    document =_list.get_ds_doc(rid, perm)
    #print "document=",document
    if document is None:
        document =_list.get_doc(rid, perm)
        if document is None:
            return HttpResponse(json.dumps({"error":"data not found"}), content_type="application/json")  
        
    filename=get_log_fname(rid,ltype)
    fsize,logtxt=get_log_content(filename,ltype,offset)

    sts="n/a"
    if document:
        sts=document.status
    
    # call from job_logs(), disabled4reader as a flag
    if disabled4reader is None:
        return logtxt
    
    ret ={"log":logtxt, "id":rid, "status": sts, "fsize":fsize} #, "linenumb":endln}
    #time.sleep(2)  
    #print "++++++++++++++++++++++++++++++++++++++++++++>>>>", ret
    return HttpResponse(json.dumps(ret), content_type="application/json")        
