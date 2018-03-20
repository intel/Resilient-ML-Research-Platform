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
from django.db.models import Q
from django.conf import settings
from django.contrib.auth.decorators import login_required
from rest_framework.response import Response
from atdml.models import Document
from atdml.forms import DocumentForm
import subprocess
import datetime
import json, time, os, sys
####import mongo library####
sys.path.append('./atdml/tasks/db')
import query_mongo
import _result, _list
import ml_serializers 

#============================================================= for predict  Web page ==================
# for web gui
def predict(request, rid, cid, msg_id, perm,disabled4reader):
    print 'in _predict.predict(), rid=', rid
    # get perm
    #uname,grp,perm,disabled4reader=get_perm(request)

    document =_list.get_ds_doc(rid, perm)
    if not document:
        return HttpResponseRedirect(reverse('atdml.views.list'))
    

    # dataset's type:
    ds_ftype=document.file_type
    msg_error=""
    msg_success=""
    tlabel=""
    #print 'hello2:', request.method

    # set message for GET
    if msg_id=="201":
        msg_success=settings.MSG_PREDICT_SUCCESS+" Id="+str(cid)
    elif msg_id=="205":
        msg_success=settings.MSG_PREDICT_APK_UPLOAD_SUCCESS+" Id="+str(cid)
    elif msg_id=="90201":
        msg_error=settings.MSG_PREDICT_FAILED
    elif msg_id=="90202":
        msg_error=settings.MSG_PREDICT_DUPLICATED
    elif msg_id=="90901":
        msg_error=settings.MSG_RECAPTCHA_FAILED
    elif msg_id and "90902" in msg_id:
        arr=msg_id.split('.')
        if len(arr)>1: # append count to the end
            msg_error=settings.MSG_UPLOAD_OVER_MAX+" "+arr[1]
        else:
            msg_error=settings.MSG_UPLOAD_OVER_MAX

    # predict action
    action_type=request.POST.get('_action_type')
    print 'action_type=', action_type
    upload_fname=""
    newdoc=None
    # for return only
    form=DocumentForm()
    
    if request.method == 'POST': # =========== =============
        print 'in predict POST'
        dns=document.db_host
        port=document.db_port
        db=document.db_db
        tbl=document.db_tbl
        hash=""
        usr=""
        pwd=""
        n_gram=document.ml_n_gram
        opt_str=document.ml_opts
        lib=document.ml_lib
        db_proj=document.db_proj if document.db_proj else ""
        pattern=document.pattern
        pca_opts=document.ml_pca_opts
        ml_feat_threshold=request.POST.get('_feat_threshold')
        if ml_feat_threshold is None or ml_feat_threshold=="":
            ml_feat_threshold=document.ml_feat_threshold
        ds_list=document.ds_list
        if pca_opts is None:
            pca_opts=""
        
        if pattern is None:
            pattern=""
        
        # find parent dataset id
        ds_id=document.train_id
        if ds_id is None or ds_id=="None" or document.option_state=="new_featuring":
            ds_id=str(rid)  # use self's feature list, if is a feature option
            #print "hihi"
        ds_id=str(ds_id)
        exe_type=request.POST.get('_file_type')
        if not exe_type is None:
            exe_type=exe_type.lower()
        emulater_config=""
        from_api=None
        
        if "apk" in  exe_type and action_type is None:
            print "_predict.predict() in apk"
            action_type='upload_predict' # for upload apk for execution from API
            from_api="y"
            
        # upload a file to predict
        if action_type == 'upload_predict':
            form = DocumentForm(request.POST, request.FILES)
            print "exe_type=",exe_type
            if form.is_valid():
                newdoc = Document(docfile = request.FILES['docfile'])
                newdoc.filename=request.FILES['docfile']
                emulater_config=request.POST.get('_emulater_config')
                pert_flag=None
                print "emulater_config=",emulater_config
                # flag for sandbox execution
                if "apk" in exe_type  :
                    if "dynamic" in exe_type:
                        newdoc.status="apk_queued"
                        newdoc.desc="has_exe_log" # flag for apk execution log
                    elif "static" in exe_type:
                        newdoc.desc="apk static" 
                    # check if static apk, 
                elif "image" in exe_type:
                    newdoc.file_type="image_predict"
                    action_type=exe_type
                    pert_flag=request.POST.get('_pert_flag')
                    
                elif document.file_type == "ensemble":
                    # special type for ensemble
                    action_type="ensemble";
                    newdoc.file_type="ensemble_predict"
            else: # form not valid ========== ====
                print 'invalid form'
                form = DocumentForm()
        elif action_type == 'hash_predict':
            hash=request.POST.get('_hash')
            if hash:
                hash=hash.lower()
            dns=request.POST.get("_dns")
            port=request.POST.get("_port")
            db=request.POST.get('_db')
            tbl=request.POST.get('_tbl')
            usr=request.POST.get('_username')
            pwd=request.POST.get('_password')
            print "_hash=", hash
            print "dns=",dns,"_db=", db
            newdoc=Document()
            newdoc.filename=hash
            upload_fname=hash
            newdoc.db_host=dns
            newdoc.db_db=db
            newdoc.db_port=port
            newdoc.db_tbl=tbl
        else: # ajax; for sample predict
            sname=request.POST.get('filename')
            #print 'sname=',sname
            idx= sname.rindex('.') 
            if idx>0:
                tlabel=sname[idx+1:].lower().strip()
                print 'label='+tlabel+"<==="
            newdoc=Document(docfile =sname)
            newdoc.filename=sname.strip()
            newdoc.true_label=tlabel

        newdoc.submitted_by=request.user.username
        newdoc.acl_list=perm
        if newdoc.file_type is None:
            newdoc.file_type="predict" # TBD
        newdoc.ml_pca_opts=pca_opts
        newdoc.ml_feat_threshold=ml_feat_threshold
         
        
        if newdoc.docfile:
            upload_fname=newdoc.docfile.name
        
        #print "docfile.name=", newdoc.docfile.name
        #print "newdoc.filename=", newdoc.filename
        print "upload_fname=", upload_fname


        #print "********************"
        newdoc.train_id=rid
        newdoc.save()

        filename=document.filename # parent filename
        fnumb=str(document.total_feature_numb)
        cid=str(newdoc.id)
        verbose="1" # default to generate feature list
        (ret, msg_id, msg)=invoke_pred_script(rid, ds_id, cid, tlabel, upload_fname, filename, fnumb, action_type, ds_ftype \
              , dns, port, db, tbl, usr, pwd, db_proj, hash, n_gram, opt_str, lib, pattern, verbose, pca_opts, exe_type, emulater_config \
              , ml_feat_threshold, ds_list=ds_list, pert_flag=pert_flag)   
        

        print "msg_id=",msg_id, ", msg="+msg
        
        # for API
        if from_api == "y":
            print "_predict.predict() in from_api:"

            newdoc=Document.objects.get(id=cid)
            wdoc ={"id":cid, "status":newdoc.status,  "pdate": newdoc.local_processed_date()
                , "by": newdoc.submitted_by, "filename": newdoc.filename, "true_label": newdoc.true_label
                , "msg": msg, "prediction":newdoc.prediction, "msg_id": msg_id
                ,"predict_val":newdoc.predict_val,"train_id":newdoc.train_id,"feat_list":""
            }
            return Response([wdoc]) # keep same format as regular pred output
        
        if request.is_ajax():
            print "Ajax predict************"
            if msg_id=="90201" or msg_id=="90205":
                print "cid=",cid,", msg_id=",msg_id
                #ret_msg=msg_error
                ret_data={"msg": msg+" Id="+str(cid)+", filename=["+newdoc.filename+"]"}
                print "ret_data",ret_data
                return HttpResponse(json.dumps(ret_data), content_type="application/json",status=400)
            #else:
            #   ret_msg=msg_success
            #print "ret_msg="+ret_msg

            newdoc=Document.objects.get(id=cid)
            ret_data ={"id":cid, "status":newdoc.status,  "pdate": newdoc.local_processed_date()
            , "by": newdoc.submitted_by, "filename": newdoc.filename, "true_label": newdoc.true_label
            , "msg": msg, "prediction":newdoc.prediction
            }


            print "json dump="+json.dumps(ret_data)
            return HttpResponse(json.dumps(ret_data), content_type="application/json")        

    elif request.method == 'GET': # =========== =============
        print 'in _predict.predict2 GET'
        param_str=document.ml_opts
        try:
            jopts=json.loads(document.ml_opts)
        except:
            jopts={}
    else: # not POST ========== ====
        print 'not post'
        
    print "echo msg_error="+msg_error, ", msg_success="+msg_success

    predictions = Document.objects.all().filter( Q(file_type__icontains="predict"), train_id=rid).order_by('-id')[0:100]
    print "pred len=",len(predictions)
    # get sample file list
    ds_id=document.train_id
    if (rid==ds_id or document.option_state=="new_featuring"): 
        ds_id=rid  # use self's feature list

    sflist=get_sfile_list(document.filename, document.id, document.file_type,ds_id); # how to get dir?
    jopts=document.ml_opts
    pca_jopts=document.ml_pca_opts
    if pca_jopts:
        pca_jopts=json.loads(document.ml_pca_opts)
    if jopts:
        jopts=json.loads(document.ml_opts)
        jopts["learning_algorithm"]=jopts["learning_algorithm"].title().replace("_"," ")

    #print "has_roc=",has_roc,", has_mrun=",has_mrun
    recaptcha=settings.RECAPTCHA_PREDICT
    if recaptcha is None:
        recaptcha="N"
        
    return render(request,
        'atdml/predict.html',
        {'document': document, 'form': form, 'predictions':predictions 
                    , 'disabled4reader':disabled4reader, 'perm':perm
                    , 'msg_error':msg_error, 'msg_success': msg_success, 'sflist':sflist
            ,"jopts":jopts,"pca_jopts":pca_jopts
            , "MEDIA_URL": settings.MEDIA_URL
            , "use_recaptcha": recaptcha
        },  
    )


#================================================= for emulator ==================
#  simplify version for collecting params for prediction 
#   classify a target doc by the model doc; used by APK emulator classification
def invoke_pred_script_by_docs(model_doc, pred_doc, action_type, ml_feat_threshold, exe_type, emulater_config):
    rid=model_doc.id
    ds_id=model_doc.train_id
    if ds_id is None or ds_id=="None" or document.option_state=="new_featuring":
        ds_id=str(rid)  # use self's feature list, if is a feature option
    cid=pred_doc.id
    tlabel="" # not used
    upload_fname=pred_doc.filename
    filename=model_doc.filename # parent filename by ATD only
    fnumb=str(model_doc.total_feature_numb)
    uploadtype=model_doc.file_type
    
    dns=model_doc.db_host
    port=model_doc.db_port
    db=model_doc.db_db
    tbl=model_doc.db_tbl
    hash=""
    usr=""
    pwd=""
    
    n_gram=model_doc.ml_n_gram
    opt_str=model_doc.ml_opts
    lib=model_doc.ml_lib
    db_proj=model_doc.db_proj if model_doc.db_proj else ""
    pattern=model_doc.pattern if model_doc.pattern else ""
    pca_opts=model_doc.ml_pca_opts if model_doc.ml_pca_opts else ""
    if ml_feat_threshold is None or ml_feat_threshold=="":
        ml_feat_threshold=model_doc.ml_feat_threshold
    ds_list=model_doc.ds_list
    verbose="1" # default to generate feature list

    return invoke_pred_script(rid, ds_id, cid, tlabel, upload_fname, filename, fnumb, action_type
        , uploadtype, dns, port, db, tbl, usr, pwd, db_proj, hash, n_gram, opt_str, lib, pattern
        , verbose, pca_opts, exe_type, emulater_config,  ml_feat_threshold, ds_list)   
    
#============================================================= predict_massive ==================
# for RESTful API only. Massive prediction without trace
def invoke_pred_script(rid, ds_id, cid, tlabel, upload_fname, filename, fnumb, action_type, ds_ftype
              , dns, port, db, tbl, usr, pwd, db_proj, hash, n_gram, opt_str, lib, pattern, verbose, pca_opts, exe_type
              , emulater_config="", ml_feat_threshold=0, ds_list=None
              , pert_flag=None):    
    print "In invoke_pred_script: rid=",rid,",ds_id=",ds_id,",cid=",cid,",upload_fname=",upload_fname,",filename=",filename
    print ",TASK_EXE=",settings.TASK_EXE,",PREDICT_SCRIPT=",settings.PREDICT_SCRIPT
    print ",fnumb=",fnumb,",tlabel="+tlabel+"<====, verbose=",verbose,",action_type=",action_type,",ds_ftype=",ds_ftype
    print ",dns=",dns,"port=",port,"db=",db,"tbl=",tbl,"db_proj=",db_proj
    print ",hash=",hash,"n_gram=",n_gram,"lib=",lib,"opt_str=",opt_str
    print ",pattern=",pattern,',pca_opts=',pca_opts,"exe_type=",exe_type , ",ds_list=",ds_list
    ret=-1   
    
    # predict single
    ret=subprocess.call([
         settings.TASK_EXE
        ,settings.PREDICT_SCRIPT # predict.sh
        ,upload_fname
        ,str(rid)
        ,filename if filename else ""
        ,str(cid)

        ,str(fnumb)
        ,action_type
        ,tlabel if tlabel else ""
        ,ds_ftype
        
        , dns if dns else ""
        , port if port else ""
        , db if db else ""
        , tbl if tbl else ""
        ,str(hash)
        ,str(n_gram) if n_gram else ""
        , opt_str if opt_str else ""
        , lib if lib else ""
        ,usr ,pwd
        ,str(db_proj) if db_proj else ""
        ,str(ds_id)
        ,pattern if pattern else ""
        ,str(verbose)
        ,pca_opts if pca_opts else ""
        ,exe_type if exe_type else ""
        ,emulater_config if emulater_config else ""
        ,str(ml_feat_threshold) if ml_feat_threshold else "0"
        ,ds_list if ds_list else ""
        ,pert_flag if pert_flag else ""
    ])
    print 'End predict script: ret=', ret, '; id=', rid,', fname=', upload_fname 
    if ret==0:
        msg_id="201"
        msg=settings.MSG_PREDICT_SUCCESS+" Id="+str(cid)
    elif ret==205: 
        msg_id="205"
        msg=settings.MSG_PREDICT_APK_UPLOAD_SUCCESS+" Id="+str(cid)
    else:
        msg_id="90201"
        msg=settings.MSG_PREDICT_FAILED   

    return (ret, msg_id, msg)
    
#============================================================= predict_massive ==================
# for RESTful API. Download data from MongoDB and massive prediction without trace.
def predict_massive(document, hash_list, host, port, db, tbl, usr, pwd, model_filename="", keep_flag="0"
        , feat_threshold=None):
    print "in _predict.predict_massive hash_list=", hash_list

    rid=str(document.id)
    ds_ftype=document.file_type
    pattern=document.pattern
    if pattern is None:
        pattern=""
        
    out_dir=settings.TRAIN_DES_DIR
    out_fname=str(datetime.now())
    out_fname="predict_output_"+out_fname.replace("-","").replace(":","").replace(" ","_")[0:19]+".json"
    out_fname=os.path.join(out_dir,rid,out_fname)
    feat_threshold=feat_threshold if feat_threshold else document.ml_feat_threshold
    
    print "rid=",rid
    print "ds_ftype=",ds_ftype,",pattern=",pattern #,",n_gram=",n_gram
    ret=-1   
    # predict multipue sample
    ret=subprocess.call([settings.TASK_EXE,
                    settings.PREDICT_MASSIVE_SCRIPT # predict_massive.sh
                    ,rid
                    ,ds_ftype     # document.file_type
                    ,out_fname
                    ,pattern
                    ,hash_list
                    ,host, str(port),db,tbl # for sample mongo db
                    ,usr, pwd
                    ,model_filename
                    ,keep_flag
                    ,str(feat_threshold)
    ])
    pred_doc=[]
    # get output file here ====================
    if ret==0:
        with open(out_fname,'r') as outf:
            pred_doc=json.load(outf)
        print "find output file here for pred_doc"
        if keep_flag =="0":
            try:
                os.remove(out_fname)
            except:
                pass
    elif ret==101:
        print "model file error. ret=101"
        pred_doc=[{"error":"model file error. ret="+str(ret)}]        
    else:
        print "find output file for error"
        pred_doc=[{"error":"please find log file for details. ret="+str(ret)}]
    return pred_doc
    
#============================================================= predict_hash for API ==================
# for RESTful API only. TBD...
def predict_hash(document, newdoc, hash, tlabel, action_type, host,port,db,tbl,usr,pwd,verbose="0"):
    print "in _predict.predict_hash(), hash=", hash

    #upload-predict    
    if newdoc.docfile:
        upload_fname=newdoc.docfile.name
        # get real file name from newdoc; may contain "/"
        if upload_fname.index("/")>0:
            upload_fname=upload_fname[upload_fname.index("/")+1:]
        #print "newdoc.docfile.name=",newdoc.docfile.name
    else:
        upload_fname=hash
        
    rid=str(document.id)
    filename=document.filename # parent filename
    fnumb="0" if document.total_feature_numb is None else str(document.total_feature_numb)
    ds_ftype=document.file_type
    cid=str(newdoc.id)
    n_gram=str(document.ml_n_gram)
    opt_str=document.ml_opts
    lib=document.ml_lib
    db_proj= "" if document.db_proj is None else str(document.db_proj)
    pattern=document.pattern
    ml_feat_threshold=document.ml_feat_threshold
    ds_list=document.ds_list
    if pattern is None:
        pattern=""
    #verbose="0" # default 0; NOT generate feature list
    ds_id=document.train_id
    print "newdoc.train_id=",newdoc.train_id, ", document.train_id=",document.train_id
    if ds_id is None or ds_id=="None" or document.option_state=="new_featuring":
        ds_id=str(rid)  # use self's feature list, if is a feature option
    ds_id=str(ds_id)    
    
    (ret, msg_id, msg)=invoke_pred_script(rid, ds_id, cid, tlabel, upload_fname, filename, fnumb, action_type, ds_ftype \
      , host, port, db, tbl, usr, pwd, db_proj, hash, n_gram, opt_str, lib, pattern, verbose, pca_opts=None, exe_type=None \
      , ml_feat_threshold=ml_feat_threshold, ds_list=ds_list)   
    print '* end predict_hash: rc=', ret, '; id=', str(rid),', fname=', upload_fname ,', verbose=',verbose
    newdoc=Document.objects.get(id=cid)
    if ret==0:
        #msg_id="201"
        #msg=settings.MSG_PREDICT_SUCCESS
        str_status=newdoc.status
    else:
        #msg_id="90201"
        #msg=settings.MSG_PREDICT_FAILED
        newdoc.status="failed"
        newdoc.save()
    
    
    #'id', 'status', 'prediction','predict_val', 'filename', 'train_id','true_label','feat_list'
    ret_data ={"id":cid, "status":newdoc.status,  "pdate": newdoc.local_processed_date()
        , "by": newdoc.submitted_by, "filename": newdoc.filename, "true_label": newdoc.true_label
        , "msg": msg, "prediction":newdoc.prediction, "msg_id": msg_id
        ,"predict_val":newdoc.predict_val,"train_id":newdoc.train_id,"feat_list":""
    }
    # load feature list for return
    if verbose=="1":
        out_file=settings.RESULT_DIR_FULL+'/'+str(rid)+'/'+str(cid)+"_feature_list.txt"
        print "feature file=",out_file
        try:
            with open(out_file) as f:
                flist = f.readlines()        
            flist.sort() 
            flist=flist[0:30]   # load first 30
            #print flist
            ret_data["feat_list"]=flist
        except:
            print "File ["+out_file+"] not found!"
        
    return ret_data


#============================================================= API get_predict ==================    
def get_predict(request, rid, hash, document, perm,disabled4reader):
    print "in _predict.get_predict, hash=",hash," user=", request.user
    
        
    if hash:
        hash=hash.lower()
   
    if request.method == 'GET':
        doc= Document.objects.all().filter(file_type__icontains="predict", train_id=rid, filename=hash)
        #slz = PredictSerializer(doc, many=True)
        #return Response(slz.data)
        return Response(ml_serializers.pred2json(doc))
    if request.method == 'POST':
        print "in POST, list=", request.POST.get('list')
        hash_list=[]
        if hash=='list':
            hash_str=request.POST.get('list')
            if hash_str:
                hash_str=hash_str.lower()
                hash_list = hash_str.split(',')
                # get unique items
                hash_list=set(hash_list)
        else: 
            hash_list.append(hash)
        host=request.POST.get('host')
        host="" if host is None else host
        port=request.POST.get('port')
        port="" if port is None else port
        db=request.POST.get('db')
        db="" if db is None else db
        tbl=request.POST.get('tbl')
        tbl="" if tbl is None else tbl
        usr=request.POST.get('usr')
        usr="" if usr is None else usr
        pwd=request.POST.get('pwd')
        pwd="" if pwd is None else pwd
        
        pred_doc=[]
        # forloop to predict each item
        for a_hash in hash_list:
            # create newdoc
            newdoc=Document()
            newdoc.filename=a_hash
            newdoc.submitted_by=request.user.username
            newdoc.acl_list=perm
            newdoc.train_id=str(rid)
            newdoc.file_type="predict" 
            newdoc.db_host=host
            newdoc.db_db=db
            newdoc.db_port=port
            newdoc.db_tbl=tbl    
            newdoc.save()
            cid=newdoc.id
            upload_fname=a_hash  
            print "before predict_hash *************** "
            #ret=_result.predict_hash(document, newdoc, a_hash, tlabel="", action_type='hash_predict'
            ret=predict_hash(document, newdoc, a_hash, tlabel="", action_type='hash_predict'
                , host=host,port=port,db=db,tbl=tbl,usr=usr,pwd=pwd)
            #ret={'rid':rid,"the_post":request.POST.get('the_post'),"username":request.user.username}
            print 'in POST: ret=', ret

            retdoc = Document.objects.get(id=cid)
            pred_doc.append(retdoc)
        return Response(ml_serializers.pred2json(pred_doc))
    else:
        ret={"error":"no support"}
        return Response(json.dumps(ret),status=404)    
#============================================================= API get_all_predicts ==================
def get_all_predicts(request,rid, perm,disabled4reader,count):
    print "in _predict.get_all_predicts, rid=",rid," user=", request.user
    if request.method == 'GET':
        doc =_list.get_ds_doc(rid, perm)
        # check permission
        if not doc:
            return Response({"error":"data not found"},status=404)
        #use "like" _icontains to get predict and ensemble_predict  
        if count is None or int(count) <=0:
            predictions = Document.objects.all().filter(Q(file_type__icontains="predict"), train_id=rid).order_by('-id')
        else: 
            predictions = Document.objects.all().filter(Q(file_type__icontains="predict"), train_id=rid).order_by('-id')[:int(count)]
        #serializer = PredictSerializer(predictions, many=True)
        #return Response(serializer.data)
        return Response(ml_serializers.pred2json(predictions))

#============================================================= get sample files for predict ==================
def get_sfile_list(filename,rid,file_type,oid):
    # for train option
    if oid:
        rid=oid # get list by oid
    print "get_sfile_list. rid=", rid
    listf=settings.RESULT_DIR_FULL+'/'+str(rid)+'/'+str(rid)+'_samplelist.txt'

    flist=[]
    try:
        with open(listf) as f:
            flist = f.readlines()        
        #print blist
        flist.sort() 
        flist=flist[0:800]   
    except:
        print "File ["+listf+"] not found!"
    return flist        
    
#============================================================= return pred row, TBD for usage ==================
def get_pred(rid, perm,disabled4reader):
    print 'in get_pred, rid=', rid
    predictions = Document.objects.all().filter(file_type__icontains="predict", id=rid)
    
    #print "t=",type(predictions),",predictions=",predictions
    
    # check if dataset doc accessible
    train_id=predictions[0].train_id
    ds_doc =_list.get_ds_doc(train_id, perm)
    if not ds_doc:
        return Response({"error":"data not found"},status=404)
    
    #serializer = PredictSerializer(predictions, many=True)
    #return Response(serializer.data)
    return Response(ml_serializers.pred2json(predictions))

#============ get pred doc by id, for delete to chk record ==================
def get_pred_doc(rid, perm,disabled4reader):
    print 'in get_pred_doc, rid=', rid
    predictions = Document.objects.all().filter((Q(file_type="emulate") | Q(file_type__icontains="predict")), id=rid)
    
    return predictions
    
