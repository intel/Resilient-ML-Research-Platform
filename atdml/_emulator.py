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
from atdml.models import Document
from atdml.forms import DocumentForm
import subprocess
import datetime
import json, time, os, sys
####import mongo library####
sys.path.append('./atdml/tasks/db')
import query_mongo
import _result, _list, _predict
from ml_serializers import *
from rest_framework.response import Response

#============================================================= emulate/predict ==================
# for web gui
def emulate(request, rid, cid, msg_id, perm,disabled4reader, from_api="n"):
    print 'in emulate, cid=', cid,", rid=", rid,",perm=",perm
    document=None
    if not rid is None and len(rid)>0:
        document =_list.get_ds_doc(rid, perm)
    
    msg_error=""
    msg_success=""
    msg_info=""
    msg_warning=""
    new_id=None

    # set message for GET
    if msg_id=="101":
        msg_success=settings.MSG_UPLOAD_SUCCESS+" Id="+str(cid)
    elif msg_id=="90101":
        msg_error=settings.MSG_UPLOAD_FAILED
    elif msg_id=="90901":
        msg_error=settings.MSG_RECAPTCHA_FAILED
    elif msg_id and "90902" in msg_id:
        arr=msg_id.split('.')
        if len(arr)>1: # append count to the end
            msg_error=settings.MSG_UPLOAD_OVER_MAX+" "+arr[1]
        else:
            msg_error=settings.MSG_UPLOAD_OVER_MAX
    exe_type=None
    recaptcha=settings.RECAPTCHA_PREDICT
    if recaptcha is None:
        recaptcha="N"


    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            desc=request.POST.get('_desc')
            emulater_config=request.POST.get('_emulater_config')
            train_id=request.POST.get('hf_train_id')
            print "desc=", desc ,",train_id=",  train_id,",perm=",perm
            # assume "<id> <type> <other info>"; append to desc for ref
            if " " in train_id:
                tarr= train_id.split(" ")
                train_id=tarr[0]
                exe_type=tarr[1].lower()
                desc=desc +", by "+ train_id+" "+exe_type
            
            newdoc = Document(docfile = request.FILES['docfile'])
            newdoc.filename=request.FILES['docfile'] #hardcode to remove "upload/"
            newdoc.submitted_by=request.user.username
            newdoc.acl_list=perm
            newdoc.file_type="emulate"     # for AE page only
            newdoc.desc=desc        # user input + ds info
            
            if not train_id is None and train_id > "":
                newdoc.status="apk_queued"  # flag "apk_queued" for prediction job          
                newdoc.train_id=train_id    # bind to a ML model for prediction
                #newdoc.file_type="predict" # predict page only
            else:
                newdoc.status="submitted"  # "submitted" for APK emulator without prediction         
                newdoc.train_id=-1         # flag to not be a dataset
            #newdoc.desc="has_exe_log" # flag for execution log
            if not rid is None:
                newdoc.train_id=rid
                # TBD for rid assigned
            newdoc.save()
            new_id=str(newdoc.id)
            
            realname=os.path.basename(newdoc.docfile.name)
            #dir_indx=realname.index(settings.UPLOAD_DIR) 
            print "realname=",realname
            print "UPLOAD_DIR=",settings.UPLOAD_DIR

            print "before Save ========="
            # filename may be different if filename duplicated
            if realname != newdoc.filename:
                newdoc.filename=realname
                newdoc.save()
            print "After Save =========="
            
            # with prediction, invoke _predict  ============
            if not train_id is None and train_id > "":
                mdoc=_list.get_shared_doc(train_id, perm)
                print "mdoc=",mdoc
                action_type='upload_predict'
                if exe_type is None:
                    exe_type="apk-dynamic"
                ml_feat_threshold=None
                # invoke predict script
                (ret, msg_id, msg)=  _predict.invoke_pred_script_by_docs( \
                    mdoc, newdoc, action_type, ml_feat_threshold \
                    , exe_type, emulater_config)
                if ret==0 or ret==205:
                    msg_success=msg
                else:
                    msg_error=msg
            else: # emulator only ============
                (ret, msg_id, msg_success, msg_error)=invoke_apk_script(realname, cid=new_id \
                , emulator_config=emulater_config)
            
        else:   #  invalid form
            # for return only
            form=DocumentForm()
    

        if from_api=="y":
            if not new_id is None:
                newdoc =_list.get_doc(new_id, perm)
            if not newdoc is None:
                msg_id="0"
                msg="APK submitted."
                retj={"id":new_id, "status":newdoc.status, "by":newdoc.submitted_by , "filename":newdoc.filename
                    , "msg_id":msg_id,"msg":msg
                }
                return Response(retj)
            else:
                return Response({"error":"submit error!"},status=404)
            
            
        # for ae_list page =============== ===
        return render(request,
            'atdml/ae_list.html',
            {'form': form, 'disabled4reader':disabled4reader, 'perm':perm
                , 'msg_error':msg_error, 'msg_success': msg_success, 'msg_info': msg_info, 'msg_warning': msg_warning
                , 'new_id': new_id #, 'options': options
                , "use_recaptcha": recaptcha
            },  
            #context_instance=RequestContext(request)
        )    

    elif request.method == 'GET': # =========== =============
        print 'in _emulator.emulate() GET'
        if from_api=="y":
            doc=None
            if not cid is None:
                doc =_list.get_doc(cid, perm)
            if not doc is None:
                msg_id="0"
                msg=""
                retj={"id":cid, "status":doc.status, "by":doc.submitted_by , "filename":doc.filename
                    , "msg_id":msg_id,"msg":msg
                }
                return Response(retj)
            return Response({"error":"record not found"},status=404)
        else:
            form=DocumentForm()

    else: # not POST ========== ====
        print 'in _emulator.emulate not post'
        
    print "msg_error="+msg_error, ",msg_success="+msg_success

    
    # for ae_list page =============== ===
    #return render_to_response(
    return render(request,
        'atdml/ae_list.html',
        {'form': form, 'disabled4reader':disabled4reader, 'perm':perm
            , 'msg_error':msg_error, 'msg_success': msg_success, 'msg_info': msg_info, 'msg_warning': msg_warning
            , 'new_id': new_id #, 'options': options
            , "use_recaptcha": recaptcha
        },  
        #context_instance=RequestContext(request)
    )    


#============================================================= invoke_apk_script ==================
# call APK shell script
def invoke_apk_script(upload_filename,cid,emulator_config=None):    
    print "In invoke_apk_script: cid=",cid,",upload_filename=",upload_filename,",cid=",cid,",emulator_config=",emulator_config
    #print ",TASK_EXE=",settings.TASK_EXE,",SETUP_APK_SCRIPT=",settings.SETUP_APK_SCRIPT

    ret=-1   
    msg_success=""
    msg_error=""
    # for apk emulator
    ret=subprocess.call([
         settings.TASK_EXE
        ,settings.SETUP_APK_SCRIPT # predict.sh
        ,upload_filename
        ,str(cid)

        ,str(emulator_config) if emulator_config else ""
    ])
    print 'End apk script: ret=', ret, '; cid=', cid,', fname=', upload_filename 
    if ret==0:
        msg_id="101"
        msg_success=settings.MSG_UPLOAD_SUCCESS+" Id="+str(cid)
    else:
        msg_id="90101"
        msg_error=settings.MSG_UPLOAD_FAILED   

    return (ret, msg_id, msg_success, msg_error)
    

    
