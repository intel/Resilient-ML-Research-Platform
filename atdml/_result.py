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
import json, time, os, sys, glob
from atdml.models import Document
from atdml.forms import DocumentForm
####import mongo library####
sys.path.append('./atdml/tasks/db')
import query_mongo
import _predict, _list


#============================================================= has_result_file ==================
def has_result_file(rid, rfile):
    rf=settings.RESULT_DIR_FULL+'/'+str(rid)+'/'+rfile
    #print settings.RESULT_DIR_FULL+'/'+str(rid)+'/'+rfile
    print "rf=",rf
    
    if '*' in rf:
        arr=glob.glob(rf)
        if arr and len(arr)>0:
            return True
        else:
            print "file=",rf,"not found!"
    elif os.path.isfile(rf):
        return True
    else:
        return False
    
#============================================================= mrun2 ==================
def mrun2(request, rid, filename, msg_id,perm,disabled4reader):
    print "in mrun2()"
    # get perm
    #uname,grp,perm,disabled4reader=get_perm(request)
    #document = Document.objects.get(id=rid)
    
    document =_list.get_ds_doc(rid, perm)
    if not document:
        return HttpResponseRedirect(reverse('atdml.views.list'))
    
    mrun_numb=""
    msg_error=""
    msg_success=""

    if msg_id=="211":
        msg_success=settings.MSG_MRUN_SUCCESS
    elif msg_id=="90211":
        msg_error=settings.MSG_MRUN_DUPLICATED

    if request.method == 'POST':

        form = DocumentForm(request.POST, request.FILES)

        action=request.POST.get('action')
        mrun_numb=request.POST.get('mrun_numb')
        print "mrun=",mrun_numb
        print '*** mrun=', action, ' rid=', rid
        # upload to HDFS ======================================================

        if document  and (action=='mrun' or action =='multiple_run'): # ================================= Multi RUN ========
            print 'In action = mrun. document.mrun_numb=', document.mrun_numb
            print '*** document.status_code=', document.status_code
            ret=0
            # only call task when number are different
            if document.mrun_numb != mrun_numb and mrun_numb:
                #update db 
                document.status='processing'
                #document.processed_date=datetime.datetime.now()
                document.mrun_numb=mrun_numb
                document.save()
                #execute shell script here
                uploadtype=document.file_type
                ml_lib=document.ml_lib
                opt_jstr=document.ml_opts
                #print document.get_file_list()

                #print settings.TASK_EXE,
                #print settings.TASK_SRC_DIR+"/"+filename
                #print "in _result.py: settings.SPARK_URL=",settings.SPARK_URL
                #print "in _result.py: settings.MRUN_SCRIPT=",settings.MRUN_SCRIPT
                
                ret=subprocess.call([settings.TASK_EXE,    #bash
                                    settings.MRUN_SCRIPT,  #multi_run.sh
                                    #settings.SPARK_SUBMIT, # spark cmd (shared)
                                    #settings.HDFS_UPLOAD_DIR+"/"+filename,  # HDFS dir for input
                                    #settings.TRAIN_DES_DIR+"/"+filename,   # dest dir
                                    rid,
                                    filename,
                                    mrun_numb,
                                    #settings.SPARK_URL,  #URL for Spark
                                    uploadtype,
                                    ml_lib,
                                    opt_jstr,
                ])
                '''
                child=subprocess.Popen([settings.TASK_EXE,
                                    settings.MRUN_SCRIPT,
                                    settings.SPARK_SUBMIT,
                                    settings.HDFS_UPLOAD_DIR+"/"+filename,
                                    settings.TRAIN_DES_DIR+"/"+filename,
                                    rid,
                                    filename,
                                    mrun_numb
                ])
                ret=child.returncode
                '''
                # refresh document
                document = Document.objects.get(id=rid)
                
                if ret==0:
                    if settings.STS_800_MRUN > document.status_code:
                        document.status='mruned'
                        document.status_code=settings.STS_800_MRUN
                        print '*** updated document.status_code=', document.status_code
                        document.processed_date=datetime.datetime.now()
                        document.save()
                        
                    print "after mrun subproc. ret=", ret
                    msg_id="211"
                else:
                    msg_id="90211"

            else: # repeated
                print "mrun repeated"
                msg_id="90212"
                
            
            print '* end mRun: rc=', ret, '; id=', rid,', fname=', filename 

            #return HttpResponseRedirect('/atdml/'+str(rid)+'/f/mrun/'+msg_id+'/')

        else: # Invalid status or action
            print '*** Invalid status or action! id=', rid,', fname=', filename 

    else: # Not POST =========
        form = DocumentForm() # A empty, unbound form

    # Load documents for the list page
    document = Document.objects.get(id=rid)
    
    predictions = Document.objects.all().filter(file_type="predict", train_id=rid).order_by('-id')[0:10]
    # get train option id
    train_id=document.train_id  
    # get sample file list
    sflist=_predict.get_sfile_list(document.filename, document.id, document.file_type, train_id); # how to get dir?
    cv_grid_data, param_str, jopts=get_cv_grid(document,rid)

    if request.is_ajax():
        print "Ajax Mrun"
        #sdoc = serializers.serializer('json', [document])
        #print "sdoc="+sdoc
        document = Document.objects.get(id=rid)
        ret_msg=""

        if msg_id=="211":
            ret_msg=settings.MSG_MRUN_SUCCESS
            ret_data ={"status":document.status, "id":rid, "pdate": document.local_processed_date()
                , "by": document.submitted_by, "vari": document.variance_short()
                , "mean": document.mean_short()
                , "msg": ret_msg+" Id="+rid, "src":mrun_numb
                    }
            return HttpResponse(json.dumps(ret_data), content_type="application/json")        
        elif msg_id=="90211": # failed
            ret_msg=settings.MSG_MRUN_FAILED
            ret_data ={"msg": ret_msg+" Id="+rid}
            print json.dumps(ret_data)
            return HttpResponse(json.dumps(ret_data), content_type="application/json",status=400)        
        elif msg_id=="90212": # duplicated
            ret_msg=settings.MSG_MRUN_DUPLICATED
            ret_data ={"msg": ret_msg+" Id="+rid}
            print json.dumps(ret_data)
            return HttpResponse(json.dumps(ret_data), content_type="application/json",status=400)        
            
        #time.sleep(2)   
    has_roc=has_result_file(rid,str(rid)+"_roc.json")
    has_mrun=has_result_file(rid,str(rid)+"_mrun.json")
    has_score=has_result_file(rid,str(rid)+"_score_graph.json")
    print "has_roc=",has_roc,", has_mrun=",has_mrun,", has_score=",has_score

    return render(request,
        'atdml/result.html',
        {'document': document, 'form': form, 'predictions':predictions
                    , 'disabled4reader':disabled4reader, 'perm':perm, 'msg_error':msg_error
                    , 'msg_success': msg_success, 'sflist':sflist
            ,"cv_grid_data":cv_grid_data,"param_str":param_str
            ,"jopts":jopts, "has_roc": has_roc, "has_mrun": has_mrun,"has_score":has_score
        },    
            #context_instance=RequestContext(request)
        )
    
    # redirect back
    #return HttpResponseRedirect(reverse('atdml.views.mrun'))


#============================================================= result ==================
def result(request, rid, perm,disabled4reader):
    print 'in result, rid=',rid
    return result2(request, rid, None, perm,disabled4reader)

def result_opts(request, rid, oid, perm,disabled4reader):
    print 'in result_opts, rid=',rid,', oid=',oid
    return result2(request, rid, oid, perm,disabled4reader)
 
def result2(request, rid, oid, perm,disabled4reader):
    print 'in result2, rid=',rid,', oid=',oid
    o_rid=rid
    # get train option doc, if oid provided
    if oid>0:
        rid=oid
    document =_list.get_ds_doc(rid, perm)
    if not document:
        return HttpResponseRedirect(reverse('atdml.views.list'))

    # for return only
    #form=DocumentForm()
    predictions = [] #Document.objects.all().filter(file_type="predict", train_id=rid).order_by('-id')[0:10]
    # get train option id
    train_id=document.train_id  
    ml_lib=document.ml_lib
    status=document.status
    # get sample file list
    sflist=_predict.get_sfile_list(document.filename, document.id, document.file_type, train_id); # how to get dir?
    # get cross validation info
    cv_grid_data, param_str, jopts=get_cv_grid(document,rid)
    print "************** ml_has_cv=",document.ml_has_cv,cv_grid_data
    
    if jopts:
        print "rid=",rid,", jopts=",jopts
    else:
        print "rid=",rid,", jopts not found"

    has_roc=has_result_file(rid,str(rid)+"_roc.json")
    has_mrun=has_result_file(rid,str(rid)+"_mrun.json")
    has_score=has_result_file(rid,str(rid)+"_score_graph.json")
    print "has_roc=",has_roc,", has_mrun=",has_mrun,", has_score=",has_score
    has_result=None
    
    # check algorithm
    train_opt={}
    if not document.ml_opts is None and len(document.ml_opts) >0:
        train_opt=json.loads(document.ml_opts)

    # 
    if document.status_code>=500:
        # check if clustering data is in
        if has_result_file(rid,str(rid)+"_cluster*.png") and train_opt["learning_algorithm"] in ('kmeans'):
            has_result="U"
        else:
            # check if png for classification exists?
            has_result="Y"
    elif ml_lib=="dnn": # allow DNN to view status
        has_result="Y"
            
    has_featc=has_result_file(rid,str(rid)+"_feat_coef.json")
    has_fp=has_result_file(rid,str(rid)+"_false_pred.json")
    
    # get ml_opts
    feature_excluded_list=None
    if "has_excluded_feat" in train_opt and train_opt["has_excluded_feat"]==1:
        # get data from mongo.dataset_info
        try:
            doc=query_mongo.find_one(settings.MONGO_OUT_DNS, settings.MONGO_OUT_PORT, settings.MONGO_OUT_DB, settings.MONGO_OUT_TBL
                , settings.MONGO_OUT_USR, settings.MONGO_OUT_PWD
                , '{"rid":'+rid+',"key":"feature_excluded"}', '{"value":1}')
            if not doc is None:
                #print "doc type=", type(doc), ",doc=",doc
                feature_excluded_list=doc["value"]  
                print "feature_excluded_list=",feature_excluded_list
        except Exception as e:
            print "Exception from MongoDB:",e
    
    rpage='atdml/result.html'
    if oid>0:
        rpage='atdml/result_opts.html'
    feat_str=""    
    if not feature_excluded_list is None:
        feat_str=','.join( str(i) for i in feature_excluded_list)
    print "has_roc=",has_roc,", has_mrun=",has_mrun,", has_result=",has_result,"rpage=",rpage
    
    # get perf and dataset info
    if document.perf_measures and document.perf_measures != "null":
        perf_measures=json.loads(document.perf_measures)
    else:
        perf_measures={}
    if document.dataset_info and document.dataset_info != "null":
        dataset_info=json.loads(document.dataset_info)
    else:
        dataset_info={}
    return render(request,
        #'atdml/result.html',
        rpage,
        {"document": document , "predictions":predictions , "sflist":sflist#, "form": form
            ,"disabled4reader":disabled4reader, "perm":perm 
            ,"cv_grid_data":cv_grid_data,"param_str":param_str,"has_fp":has_fp
            ,"jopts":jopts, "has_roc": has_roc, "has_mrun": has_mrun, "has_result":has_result, "has_featc":has_featc,"has_score":has_score
            ,"feature_excluded": feat_str,"ml_lib":ml_lib, "status":status
            , "tp": perf_measures["tp"] if "tp" in perf_measures  else ""
            , "tn": perf_measures["tn"] if "tn" in perf_measures  else ""
            , "fp": perf_measures["fp"] if "fp" in perf_measures  else ""
            , "fn": perf_measures["fn"] if "fn" in perf_measures  else ""
            , "phi":'%0.5f'%  perf_measures["phi"] if "phi" in perf_measures  else ""
            , "fscore":'%0.5f'%  perf_measures["fscore"] if "fscore" in perf_measures  else ""
            , "roc_auc":'%0.5f'%  perf_measures["roc_auc"] if "roc_auc" in perf_measures  else ""
            , "class_count": dataset_info["class_count"] if "class_count" in dataset_info  else ""
            , "training_fraction": dataset_info["training_fraction"] if "training_fraction" in dataset_info  else ""
            , "dataset_count": dataset_info["dataset_count"] if "dataset_count" in dataset_info  else ""
            , "MEDIA_URL": settings.MEDIA_URL
        }, 
    ) 

    
# get cv_grid, param str & jopts
#============================================================= get_cv_grid ==================
def get_cv_grid(document,rid):
    cv_grid_data=None
    param_str=None
    jopts=None
    doc=None

    if document.ml_has_cv=="yes":
        # get data from mongo.dataset_info
        try:
            doc=query_mongo.find_one(settings.MONGO_OUT_DNS, settings.MONGO_OUT_PORT, settings.MONGO_OUT_DB, settings.MONGO_OUT_TBL
                , settings.MONGO_OUT_USR, settings.MONGO_OUT_PWD
                , '{"rid":'+rid+',"key":"cv_result"}', '{"param_str":1,"cv_grid_data":1,"best_param":1,"_id":0}')
        except Exception as e:
            print "Exception from MongoDB:",e

        # if CV exists
        if doc:
            #print "cv_grid_data=",str(doc["cv_grid_data"])
            cv_grid_data=doc["cv_grid_data"]         
            param_str=doc["param_str"] 
    # convert ml_opts to json object, and set 1st uppercase & remove _
    if doc and document.ml_opts:
        jopts=json.loads(document.ml_opts)
        jopts["learning_algorithm"]=jopts["learning_algorithm"].title().replace("_"," ")
            
    return cv_grid_data, param_str, jopts

    
    