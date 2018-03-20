#!/usr/bin/python
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
from rest_framework.response import Response
import subprocess, datetime
import json, time, os, sys
from atdml.models import *
from atdml.forms import DocumentForm
import _result, _list

####import mongo library####
sys.path.append('./atdml/tasks/db')
import query_mongo

FILTER_COUNT=settings.FEATURE_IMPO_FILTER_COUNT
LIST_COUNT=settings.FEATURE_IMPO_LIST_COUNT
MAX_DISPLAY_LEN=50


#============================================================= feature_impo for combined file ==================
# msg_id: success=231
#         fail=
@login_required
def feature_impo(request,rid, perm,disabled4reader):
    return feature_impo2(request,rid, perm,disabled4reader,"")
    
def feature_impo2(request,rid, perm,disabled4reader,msg_id):
    print "in feature_impo2(), rid=",rid
    msg_success=""
    msg_error=""
    #msg_id for message after POST and avoid re-POST
    if msg_id=="231":
        msg_success=settings.MSG_FEATURE_IMPO_SUCCESS+" Id="+str(rid)
    elif msg_id=="232":
        msg_success=settings.MSG_FEATURE_SET_SUCCESS+" Id="+str(rid)
    elif msg_id=="233":
        msg_success=settings.MSG_FEATURE_DROP_SUCCESS+" Id="+str(rid)
    elif msg_id>="90000":
        msg_error=settings.MSG_FEATURE_IMPO_FAILED+" Id="+str(rid)
        
    document =_list.get_ds_doc(rid, perm)
    if not document:
        return HttpResponseRedirect(reverse('list'))
    
    filename=None    
    if document:
        filename=document.filename
    
    if not filename:
        msg_error="Dataset not found! id="+rid
        msg_id="90000"
        print "msg_error=",msg_error

    #get verified features from db
    try: 
        vflist = Feature_click.objects.all().filter(rid=rid,vote__gte=FILTER_COUNT).order_by('-vote')[:LIST_COUNT]
    except : 
        vflist = []
    

    #get combined features from file
    out_COMB=settings.RESULT_DIR_FULL+"/"+str(rid)+"/"+str(rid)+"_score_combine.txt"

    lines =[]
    outlist1 =[]
    items=[]
    no_feature="N"
    try:
        with open(out_COMB,'r') as f:
            lines=f.read().splitlines()
    except:
        no_feature="Y"
        pass
    
    if len(lines)>=LIST_COUNT:
        lines=lines[:LIST_COUNT]

    # replace \t with html tag
    for idx, line in enumerate(lines):
        #line = line.replace('\t','</td><td>')
        items = line.split('\t')
        # chk box, fid, score, desc, id
        line='<td ><input type="checkbox" class="checkbox" name="ck_fid" value="'+items[0]+'"></input></td>' \
                +'<td>'+items[0]+'</td><td>'+items[1]+'</td><td data-placement="bottom" data-toggle="tooltip" ' \
				+' title="'
        # escape <  and > for html
        items[2]=items[2].replace("<","&lt;").replace(">","&gt;")
        if len(items[2])>MAX_DISPLAY_LEN:
            line+=items[2]+'">'+show_partial(items[2])
        else:
            line+='">'+items[2]           
        line+='</td><td>'+str(idx+1)+'</td>'
        outlist1.append(line)
    jopts=document.ml_opts
    if jopts:
        jopts=json.loads(document.ml_opts)
        jopts["learning_algorithm"]=jopts["learning_algorithm"].title().replace("_"," ")

    return render(request,
        'atdml/feature.html',
        {'document': document, 'vflist':vflist, 'flist1':outlist1, 'msg_success': msg_success, 'msg_error': msg_error
            , 'no_feature': no_feature, 'jopts':jopts }, 
        #context_instance=RequestContext(request)
    ) 
#============================================================= show partial string ==================
def show_partial(str):
    l=len(str)
    idx=int(MAX_DISPLAY_LEN/2)
    if l> MAX_DISPLAY_LEN:
        return str[:idx]+".. .."+str[l-idx-4:l]
    return str    

#============================================================= exclude_feature ==================
# for result page to exculde feature from training
#         fail=
@login_required
def exclude_feature(request,rid, perm,disabled4reader):
    print "in exclude_feature"
    msg_id=None
    document =_list.get_ds_doc(rid, perm)
    if not document:
        return HttpResponseRedirect(reverse('list'))
        
    # find parent dataset id    
    train_id=document.train_id
    if train_id is None:
        train_id=rid
    
    excl_feat=None
    if request.method == 'POST': 
        excl_feat=request.POST.get('hf_w_excl_feat')
         
    print "excl_feat=",excl_feat

    json2save={}
    fid_dict={}
    ml_opts=json.loads(document.ml_opts)
    has_excl_key=0
    
    # check if key exists
    if "has_excluded_feat" in ml_opts:
        has_excl_key=1
    
    print "ml_opts=",ml_opts," type=", type(ml_opts),", excl_feat=",excl_feat
    fid_arr=[]
    # build dict for excluded feature
    if not excl_feat is None and len(excl_feat)>0:
        fid_arr=excl_feat.split(',')

        # update ml_opts
        ml_opts["has_excluded_feat"]=1
        has_excl_key=1
    else: # excl feat was removed
        if has_excl_key==1:
            ml_opts["has_excluded_feat"]=0

    # only update if excl key exists
    if has_excl_key==1:
        # update ml_opts
        document.ml_opts=json.dumps(ml_opts)
        #print  "ml_opts str=",json.dumps(ml_opts)
        document.save()

        # save exclude list to mongo        
        json2save["rid"]=eval(rid)
        json2save["key"]="feature_excluded"
        json2save["value"]=fid_arr

        feat_excl=json.dumps(json2save)
        #print "feature_excluded=",feat_excl
        filter='{"rid":'+rid+',"key":"feature_excluded"}'
        upsert_flag=True
        #print "filter=",filter,",feat_excl=",feat_excl
        ## write to mongoDB.myml.dataset_info, ignore doc with duplicated key
        ret=query_mongo.upsert_doc(settings.MONGO_OUT_DNS, settings.MONGO_OUT_PORT, settings.MONGO_OUT_DB, settings.MONGO_OUT_TBL
            , settings.MONGO_OUT_USR, settings.MONGO_OUT_PWD
                ,filter,feat_excl,upsert_flag)
        print "Upsert count for feat_excl: ret=",ret    
    
    return HttpResponseRedirect(reverse('result_opts',args=[train_id,rid]))
    
    
#============================================================= set_feature ==================
# msg_id: success=232, 233
#         fail=
# update Feature_click.vote
@login_required
def set_feature(request,rid, perm,disabled4reader):
    #print "in set_feature"
    msg_id=None
    document =_list.get_ds_doc(rid, perm)
    if not document:
        return HttpResponseRedirect(reverse('list'))
    
    filename=document.filename
    uploadtype=document.file_type

    #get files
    ck_list = None
    feat=None
    #print ck_list
    if request.method == 'POST': 
        action=request.POST.get('action')
        #print 'in post action=', action
        has_change=0
        
        if action=='vote_fid':
            ck_list =request.POST.getlist('ck_fid')
            to_verified =request.POST.getlist('to_verified')
            #print 'to_verified=',to_verified
            # add a new one
            for idx, fid in enumerate(ck_list):
                feat=None
                try: 
                    if not Feature_click.objects.all().filter(fid=fid, rid=rid):
                        feat = Feature_click(fid = fid,rid=rid, vote=1)
                        if len(to_verified)==1 and to_verified[0]=="1":
                            feat.vote=FILTER_COUNT
                    else: # increase vote count
                        feat=Feature_click.objects.get(fid=fid, rid=rid)
                        if len(to_verified)==1 and to_verified[0]=="1" and feat.vote <FILTER_COUNT:
                            feat.vote=FILTER_COUNT
                        else:    
                            feat.vote=feat.vote+1
                    feat.save()
                    has_change=1
                except : 
                    feat = None
                    #error msg?
                    
            # call feature_impo API to refresh combine list
            if has_change ==1:
                ds_id=None
                if document.option_state == "new_featuring":
                    # having featuring output, not depends on source dataset id.
                    ds_id=document.train_id
                ret=invoke_feature_impo(filename, rid, uploadtype,"comb_only",ds_id)
            
            #check ret?
                
            msg_id="232" # success msg
        # drop feature list item ===========================================
        if action=='drop_fid':
            ck_list =request.POST.getlist('vf_fid')
            #print ck_list
            # add a new one
            for idx, fid in enumerate(ck_list):
                feat=None
                try: 
                    if Feature_click.objects.all().filter(fid=fid, rid=rid):
                        feat = Feature_click.objects.get(fid=fid, rid=rid)
                        feat.vote=0 # reset to 0
                        feat.save()
                except : 
                    feat = None
                    #error msg?
                
            msg_id="233" # success msg
    else: # not POST ========== ====
        print 'invalid method'

 

    return feature_impo2(request,rid, perm,disabled4reader, msg_id) 


#============================================================= calculate_feature_impo ==================
# msg_id: success=232, 233
#         fail=
# update Feature_click.vote
@login_required
def calculate_feature_impo(request,rid, perm,disabled4reader):
    print 'In calculate_feature_impo'
    document =_list.get_ds_doc(rid, perm)
    if not document:
        return HttpResponseRedirect(reverse('list'))
    
    filename=document.filename
    uploadtype=document.file_type
    document.status='processing feature importance'
    #document.processed_date=datetime.datetime.now() why failed?
    document.processed_date=datetime.now()
    document.save()
    
    ds_id=None
    if document.option_state == "new_training":
        # having featuring output, not depends on source dataset id.
        ds_id=document.train_id
    else:
        ds_id=rid
    
    # call feature_impo API
    ret=invoke_feature_impo(filename, rid, uploadtype,"",ds_id)
    #execute shell script here

    
    print "feat import ret=", ret
    # update status code
    document = Document.objects.get(id=rid)
    msg_id=-1
    if ret ==0 :
        document.status='importance_calculated'
        if settings.STS_1000_FEATURE_IMPO > document.status_code:
            document.status_code=settings.STS_1000_FEATURE_IMPO
        msg_id="231" 
    else:
        document.status='feature importance failed'    
        msg_id="90231" 
    #document.processed_date=datetime.datetime.now()
    document.processed_date=datetime.now()
    document.save()    
    
    print '* end Feature importance: rc=', ret, '; id=', rid

    return msg_id # _list to handle return page
    
#============================================================= invoke_feature_impo ==================
def invoke_feature_impo(filename, rid, uploadtype, ctype, ds_id):
    
    #files for feature importance
    out_FIRM=settings.RESULT_DIR_FULL+"/"+str(rid)+"/"+str(rid)+"_score_FIRM.txt"
    out_PROB=settings.RESULT_DIR_FULL+"/"+str(rid)+"/"+str(rid)+"_score_PROB.txt"
    out_IT=settings.RESULT_DIR_FULL+"/"+str(rid)+"/"+str(rid)+"_score_IT.txt"
    out_COMB=settings.RESULT_DIR_FULL+"/"+str(rid)+"/"+str(rid)+"_score_combine.txt"

    ret=-1
    #print 'feature_impo call '
    ret=subprocess.call([settings.TASK_EXE,         #bash
                        settings.FEATURE_IMPO_SCRIPT,      #feature_importance.sh
                        rid,
                        filename,
                        uploadtype,
                        out_FIRM,  # dest dir+ filename
                        out_PROB,
                        out_IT,
                        out_COMB,
                        ctype,
                        str(ds_id),
    ])
    print 'end call, ret=', ret
    return ret
        
#============================================================= feature_impo_all 3 files ==================
# msg_id: success=
#         fail=
@login_required
def feature_impo_all(request,rid, perm,disabled4reader):
    #print 'in feature_impo_all'

    document =_list.get_ds_doc(rid, perm)
    if not document:
        return HttpResponseRedirect(reverse('list'))

    filename=document.filename

    #get files
    out_FIRM=settings.RESULT_DIR_FULL+"/"+str(rid)+"/"+str(rid)+"_score_FIRM.txt"
    out_PROB=settings.RESULT_DIR_FULL+"/"+str(rid)+"/"+str(rid)+"_score_PROB.txt"
    out_IT=settings.RESULT_DIR_FULL+"/"+str(rid)+"/"+str(rid)+"_score_IT.txt"

    flist1,flist2,flist3 = get_feat_importance(rid,out_FIRM,out_PROB, out_IT);
    outlist1=[]
    outlist2=[]
    outlist3=[]
    # replace \t with html tag; TBD to use bootstrap table...
    for line in flist1[:LIST_COUNT]:
        out=""
        for idx, item in enumerate(line.split('\t')):
            item=item.replace("<","&lt;").replace(">","&gt;")
            print idx, item
            if idx==2:
                out += '<td data-placement="bottom" data-toggle="tooltip" ' \
                    +' title="'+item+'">'+show_partial(item)+'</td>'
            elif idx==0:
                out += '<td class="'+item+'">'+item+'</td>'
            else:
                out += '<td>'+item+'</td>'
        print "out=",out
        outlist1.append(out)
    for line in flist2[:LIST_COUNT]:
        out=""
        for idx, item in enumerate(line.split('\t')):
            item=item.replace("<","&lt;").replace(">","&gt;")
            print idx, item
            if idx==2:
                out += '<td data-placement="bottom" data-toggle="tooltip" ' \
                    +' title="'+item+'">'+show_partial(item)+'</td>'
            elif idx==0:
                out += '<td class="'+item+'">'+item+'</td>'
            else:
                out += '<td>'+item+'</td>'
        print "out=",out
        outlist2.append(out)        
    for line in flist3[:LIST_COUNT]:
        out=""
        for idx, item in enumerate(line.split('\t')):
            item=item.replace("<","&lt;").replace(">","&gt;")
            print idx, item
            if idx==2 and len(item)>MAX_DISPLAY_LEN:
                out += '<td data-placement="bottom" data-toggle="tooltip" ' \
                    +' title="'+item+'">'+show_partial(item)+'</td>'
            elif idx==0:
                out += '<td class="'+item+'">'+item+'</td>'
            else:
                out += '<td>'+item+'</td>'
        print "out=",out
        outlist3.append(out)        #line = line.replace('\t','</td><td>')
        
    is_option='N'
    if document.train_id:
        is_option='Y'
    jopts=document.ml_opts
    if jopts:
        jopts=json.loads(document.ml_opts)
        jopts["learning_algorithm"]=jopts["learning_algorithm"].title().replace("_"," ")
        
    return render(request,
        'atdml/feature_all.html',
        {'document': document,  'flist1':outlist1, 'flist2':outlist2, 'flist3':outlist3,'is_option':is_option
            ,"jopts":jopts}, 
    ) 

#============================================================= feature_impo_combs  ==================
# msg_id: success=
#         fail=
@login_required
def feature_impo_combs(request,rid, perm,disabled4reader):
    #print 'in feature_impo_all'

    document =_list.get_ds_doc(rid, perm)
    if not document:
        return HttpResponseRedirect(reverse('list'))

        
    is_option='N'
    if document.train_id:
        is_option='Y'
    jopts=document.ml_opts
    if jopts:
        jopts=json.loads(document.ml_opts)
        jopts["learning_algorithm"]=jopts["learning_algorithm"].title().replace("_"," ")
        
    return render(request,
        'atdml/feature_combs.html',
        {'document': document,'is_option':is_option
            ,"jopts":jopts}, 
    ) 
    
    
#============================================================= get_feat_importance for API ==================
def get_feat_impo(request, rid, perm,disabled4reader):
    # chk access
    document =_list.get_ds_doc(rid, perm)
    if not document:
        return Response({"data not found":-1})
    
    # get data from mongo.dataset_info
    doc=query_mongo.find_one(settings.MONGO_OUT_DNS, settings.MONGO_OUT_PORT, settings.MONGO_OUT_DB, settings.MONGO_OUT_TBL
        , settings.MONGO_OUT_USR, settings.MONGO_OUT_PWD
        , '{"rid":'+rid+',"key":"feature_importance"}', '{"value":1,"_id":0}')
        
    if doc:
        arr=doc["value"]    
        return Response(arr)
    else:
        return Response({"data not found":-1})
	
#============================================================= get_feat_importance 3 files ==================
def get_feat_importance(rid,filenameF,filenameP,filenameI):

    lines1 =[]
    lines2 =[]
    lines3 =[]
    try:
        with open(filenameF,'r') as f:
            lines1=f.read().splitlines()
        with open(filenameP,'r') as f:
            lines2=f.read().splitlines()
        with open(filenameI,'r') as f:
            lines3=f.read().splitlines()
    except:
        pass
    return lines1,lines2,lines3

