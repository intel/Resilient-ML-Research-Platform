'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''
from django.shortcuts import render_to_response 
from django.template import RequestContext
from django.http import HttpResponseRedirect, HttpResponse
from django.http import JsonResponse
from django.http import StreamingHttpResponse
from django.core.urlresolvers import reverse
from django.core import serializers
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Q
#from django.core.servers.basehttp import FileWrapper
from rest_framework.response import Response
from atdml.models import *
from atdml.forms import DocumentForm
import subprocess, datetime
from wsgiref.util import FileWrapper
import mimetypes

import json, time, os, string, random, sys, ConfigParser, csv, glob

import ml_serializers
import _result, _feature, _list, _predict, _log, _emulator

####import our own library####
sys.path.append('./tasks/db')
import query_mongo
sys.path.append('./atdml/tasks/ml')
import ml_util

CONF_FILE='./app.config' # at the base dir of the web
config=ConfigParser.ConfigParser()
config.read(CONF_FILE)

#============================================================= get_dataset_list for list page ==================
def get_dataset_list(request, perm,disabled4reader):
    print "in _api.get_dataset_list"
    documents =_list.get_ds_doclist(perm)
    if documents and len(documents)>0:

        return Response(ml_serializers.doc2json(documents))
    else:
        #return Response({"error":"data not found"},status=404)
        return Response({"warning":"data not found!"})

 
#============================================================= get_eslist for ensemble selector ==================
def get_eslist(request, perm,disabled4reader):        
    print "in _api.get_dataget_eslistset_list"
    documents =_list.get_docs4ensemble(perm)
    if documents and len(documents)>0:
        return Response(ml_serializers.doc2json(documents))
    else:
        #return Response({"error":"data not found"},status=404)
        return Response({"warning":"data not found!"})
        
#============================================================= get_apk_list for android emulator records ==================
def get_apk_list(request, perm,disabled4reader):        
    if perm=="5": # admin get all APK; return last 1000 items only
        documents = Document.objects.all().filter(Q(file_type="emulate") | Q(desc="has_exe_log") ).order_by('-id')[0:1000]
    else:
        documents = Document.objects.all().filter( (Q(file_type="emulate") | Q(desc="has_exe_log")), acl_list=perm).order_by('-id')[0:1000]
    if documents and len(documents)>0:

        ddict=ml_serializers.doc2json(documents)
        for v in ddict:
            v["has_zip"]=check_download_file(str(v["id"]))
        return Response(ddict)
    else:
        return Response([])
        
#============================================================= api_get_feature_impo==================
def get_feature_impo(request, rid, perm,disabled4reader):
    print "in get_feature_impo, user=", request.user
    return _feature.get_feat_impo(request, rid, perm,disabled4reader)

#============================================================= api_get_all_predicts ==================
def get_all_predicts(request,rid, perm,disabled4reader, count=0):
    print "in _api.get_all_predicts"
    return _predict.get_all_predicts(request,rid, perm,disabled4reader, count)

#============================================================= api_get_pred_stat ==================
def get_pred(request,rid, perm,disabled4reader):
    print "in _api.get_pred. rid=", rid
    if request.method == 'GET':    
        return _predict.get_pred(rid, perm,disabled4reader)
    return Response({"error":"data not found"},status=404)

    
#============================================================= api_get_ds_info ==================
# tbd? used by?
def get_ds_info(request,rid, perm,disabled4reader):
    print "in _api.get_ds_info. rid=", rid
    if request.method == 'GET':
        #doc = Document.objects.get(id=rid)
        doc =_list.get_ds_doc(rid, perm)
        if not doc:
            return Response({"error":"data not found"},status=404)
        local_processed_date=doc.local_processed_date()

        arr=[]
        arr.append([doc.id,doc.filename,doc.file_type,doc.status,local_processed_date
            , doc.desc])
        ret={}
        ret["data"]=arr

        return JsonResponse(ret, safe=False)
        
#============================================================= api_get_optlist ==================
# for opts tbl ajax call. expect a json with "data":[jdoc1,jdoc2,...]
def get_optlist(request,rid, perm,disabled4reader):
    print "in _api.get_optlist. rid=", rid
    if request.method == 'GET':
        #doc = Document.objects.get(id=rid)
        doc =_list.get_ds_doc(rid, perm)
        if not doc:
            #print "not found!"
            #return Response({"data not found":-1},status=404)
            return Response({"error":"data not found"},status=404)
        arr=[]
        # get dataset row
        arr.append(get_row_4_opt(doc))
        #print "arr=", arr
        #documents = Document.objects.all().filter(~Q(file_type='predict'),acl_list__lte=perm, train_id=rid).order_by('-id')[0:500]
        #print "before doc"
        # get option rows for this dataset
        documents = _list.get_opt_docs(rid,perm)
        #print "here, len=",len(documents)
        if documents:
            for doc in documents:
                #print "doc=",doc
                arr.append(get_row_4_opt(doc))

        jobj={}
        jobj["data"]=arr
        #print "jobj=",jobj
        return JsonResponse(jobj, safe=False)

#================== add icon and special tag for datatable row
def get_row_4_opt(doc):
    ret_json=_list.get_ret_json(doc, "")
    #icon for table, moved to javascript
    #ret_json["*"]='<img id="_img_'+str(doc.id) \
    #    +'" class="img_checked_grey _img_ds_r" style="max-height:20px; max-width:20px; " src="/static/img/checked_grey.png" />' 
    # special tag for setting id for datatable row
    ret_json["DT_RowId"]=doc.id
    return ret_json

#============================================================= api_get_model ==================
# GET model for offline prediction
# id, filename, file_type, status, local_processed_date, ml_n_gram, ml_lib, ml_opts
#   , accuracy, train_id, option_state
#   , dic_seq_hashes, coef_arr, coef_intercept, pca_param, dic_hash_str, dic_name_label
def get_model(request, rid, perm, disabled4reader):
    print "in get_model, rid=",rid
    # check permission
    document =_list.get_ds_doc(rid, perm)
    if not document:
        return Response({"error":"data not found"},status=404)
    # get model dict    
    
    local_processed_date=document.local_processed_date()
    ret={}
    ret["id"]=document.id
    ret["filename"]=document.filename
    ret["file_type"]=document.file_type
    ret["status"]=document.status
    ret["local_processed_date"]=local_processed_date
    ret["ml_n_gram"]=document.ml_n_gram
    ret["ml_lib"]=document.ml_lib
    ret["ml_opts"]=json.loads(document.ml_opts)
    ret["accuracy"]=document.accuracy    
    ret["train_id"]=document.train_id
    ret["option_state"]=document.option_state  

    # get other info from mongo
    ret=ml_util.ml_get_model(ret)

    return JsonResponse(ret, safe=False)


#============================================================= api_get_predict==================
# GET prediction(s) or POST new prediction
def get_post_predict(request, rid, hash, perm,disabled4reader):
    print "in get_post_predict, hash=",hash," user=", request.user
    # check permission
    document =_list.get_ds_doc(rid, perm)
    if not document:
        #print "not found!"
        #ret={"error":"data model not found!"}
        return Response({"error":"dataset not found"},status=404)

    if hash:
        hash=hash.lower()
   
    if request.method == 'GET':
        print "In GET: rid=",rid,",hash=",hash
        # by prediction
        if hash.isdigit():
            doc= Document.objects.all().filter(file_type="predict", train_id=rid, id=hash)
        else:
            doc= Document.objects.all().filter(file_type="predict", train_id=rid, filename=hash)
        print "doc=",doc
        # get by md5/filename
        if len(doc)>0:
            #slz = PredictSerializer(doc, many=True)
            #return Response(slz.data)
            return Response(ml_serializers.pred2json(doc))
        return Response({"error":"prediction not found"},status=404)    
        
    action_type='hash_predict'
    
    if request.method == 'POST':
        print "in POST, list=", request.POST.get('list')
        verbose=request.POST.get('verbose')
        verbose="0" if verbose is None else verbose
        print "verbose=",verbose

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
        model_filename=request.POST.get('model_filename')
        model_filename="" if model_filename is None else model_filename
        keep_flag=request.POST.get('keep_flag')
        keep_flag="0" if keep_flag is None else keep_flag
        
        predict_list=[]
        pred_doc=[]
        
        # for offline massive prediction
        if hash=='list_offline':
            hash_list=request.POST.get('list')
            feat_threshold=request.POST.get('feat_threshold')
            pred_doc= _predict.predict_massive(document, hash_list
                , host=host,port=port,db=db,tbl=tbl,usr=usr,pwd=pwd, model_filename=model_filename, keep_flag=keep_flag
                , feat_threshold=feat_threshold )
            return Response(pred_doc)
            
        # for ONE hash list
        elif 'list' in hash:
            hash_str=request.POST.get('list')
            if hash_str:
                hash_str=hash_str.lower()
                predict_list = hash_str.split(',')
                # get unique items
                predict_list=set(predict_list)

        # upload raw data for prediction
        elif 'raw' in hash:
            form = DocumentForm(request.POST, request.FILES)
            if form.is_valid():
                print "in API upload predict"
                newdoc = Document(docfile = request.FILES['docfile'])
                newdoc.filename=request.FILES['docfile']
                if document.file_type == 'ensemble': # upload binary for ensemble predict
                    action_type='ensemble_predict'
                    print "for ensemble predict..."
                else:
                    action_type='upload_predict'
                predict_list.append(newdoc.filename)
                print "newdoc.filename=",newdoc.filename
            else:
                print "Form is invalid!"
                return Response({"Error":"invalid form"},status=404)   
    
        # upload binary for sandbox execution & predict
        elif hash=='exec':
            form = DocumentForm(request.POST, request.FILES)
            if form.is_valid():
                print "in API upload for execution & wait:"
                exe_type=request.POST.get('_file_type')
                # handle by _predict; key field is "_file_type"
                if exe_type is None:
                    print "required field not found!"
                    return Response({"Error":"required field not found."},status=404)
                else:
                    return _predict.predict(request, rid, cid=None, msg_id=None, perm=perm,disabled4reader=disabled4reader)

            else:
                print "Form is wrong!!"
                return Response({"Error":"invalid form"},status=404)
        else: 
            predict_list.append(hash)
        
        # TBD need to check upload count here?
        for p_item in predict_list:
            # create newdoc
            if action_type=="hash_predict":
                newdoc=Document()
                newdoc.filename=p_item
            newdoc.submitted_by=request.user.username
            newdoc.acl_list=perm
            newdoc.train_id=str(rid)
            if action_type =="ensemble":
                newdoc.file_type="ensemble_predict"             
            else:
                newdoc.file_type="predict" 
            
            newdoc.ml_n_gram=document.ml_n_gram
            newdoc.ml_opts=document.ml_opts
            newdoc.ml_lib=document.ml_lib
                
            newdoc.db_host=host
            newdoc.db_db=db
            newdoc.db_port=port
            newdoc.db_tbl=tbl  
            
            newdoc.save()
            cid=newdoc.id
            #upload_fname=p_item  
            #print "before predict_hash *************** "                     
            ret=_predict.predict_hash(document, newdoc, p_item, tlabel="", action_type=action_type
                , host=host,port=port,db=db,tbl=tbl,usr=usr,pwd=pwd, verbose=verbose )
            #print 'in POST: ret=', ret
            pred_doc.append(ret)
                        

        return Response(pred_doc)
    else:
        return Response({"error":"data not found"},status=404)

#============================================================= get_log_file for text download ==================
# generic function to get text file in log folder
def get_log_file(rid, ltype, offset, perm,disabled4reader):
    print 'in get_log_file(), rid=', rid, ",offset=", offset
    # check access
    document =_list.get_ds_doc(rid, perm)
    #print "document..=",document
    if document is None:
        # check if record exist
        document =_list.get_doc(rid, perm)
        if document is None:
            return Response({"error":"file not found"})
    
    return _log.get_log_file(rid, ltype, offset, perm,disabled4reader)
#============================================================= download_file ==================
# generic function to download file; for downloading APK from AWS
def download_file(rid, fname, perm, disabled4reader, type="apk_zip"):
    print 'in download_file(), rid=', rid
    # check access
    if not rid is None:
		# check if record exist, exclude "__apk__" type
        document =_list.get_doc(rid, perm)
        if document is None and not type in ("__apk__"):
            return Response({"error":"file not found"})
    RLT_DIR=settings.EXEC_RESULT_FOLDER #'/home/nfs/result'
    ARC_DIR=settings.EXEC_LOG_FOLDER  #"/home/nfs/archive"

    # for apk output.zip
    if type in ("apk_zip","__apk__"):
        filename=check_download_file(rid, fname, type)
        print "filename=",filename
        if not filename is None:
            if type in ("apk_zip"):
                wrapper = FileWrapper(file(filename))
                response = HttpResponse(wrapper, content_type='application/zip')
                ft=os.path.basename(filename)
            elif type in ("__apk__"):
                chunk_size = 8192
                response = StreamingHttpResponse(FileWrapper(open(filename, 'rb'), chunk_size),
                           content_type=mimetypes.guess_type(filename)[0])            
                #response = HttpResponse(wrapper, content_type='application/x-binary')
                ft=os.path.basename(filename)+"---"+ str(rid) + "---" \
                    + datetime.datetime.utcfromtimestamp(os.stat(filename)[-2]).strftime("%Y-%m-%d %H:%M:%S")
            response['Content-Disposition'] = 'attachment; filename="'+ft+'"'
            response['Content-Length'] = os.path.getsize(filename)
            return response
        else:
            return Response({"error":"file not found"})
        
    else:
        return Response({"error":"file not found"})
    # download by curl -X GET http://ip:port/atdml/api/df/__apk__/ -u id:pwd -O -J

    
#============================================================= upload_file ==================
# upload file, unzip and save; for emulator output to AWS
def upload_file(request, rid, fname=None, perm=None, disabled4reader=None, type="__apk__"
    , dest_dir=settings.EXEC_RESULT_FOLDER):
    print 'in upload_file(), rid=', rid
    # check access
    if not rid is None:
        document =_list.get_doc(rid, perm)
        if document is None and not type in ("__apk__"):
            # check if record exist
            document =_list.get_doc(rid, perm)
            if document is None:
                return Response({"error":"rid not found"})
    
    # get filename
    fname=str(request.FILES['docfile'])
    fullfname=os.path.join(dest_dir,fname)
    folder=os.path.join(dest_dir,rid)
    
    # save to file
    try:
        handle_uploaded_file(fullfname, request.FILES['docfile']) 
        print "upload file saved."
    except:
        return Response({"error":"file saving exception"},status=404)
        
    try:
        if type in ("__apk__"):
            # upzip the file
            print "folder=",folder
            if not os.path.exists(folder):
                os.makedirs(folder)
            # unzip and remove zip file
            cmd="unzip -jqo '"+fullfname+"' -d '"+folder+"'  && rm -f '"+fullfname+"'" 
            os.system(cmd)

        return Response({"msg":"file upload successfully"})
    except:
        return Response({"error":"file uzip failed"},status=404)

#============================================================= handle_uploaded_file ==================
def handle_uploaded_file(fname, data):
    with open(fname, 'wb') as dest:
        for chunk in data.chunks():
            dest.write(chunk)  
    
#============================================================= check_download_file==================
# generic function to download file
def check_download_file(rid, fname="output.zip", type="apk_zip"):
    # for apk output.zip
    RLT_DIR=settings.EXEC_RESULT_FOLDER #'/nfs/result'
    ARC_DIR=settings.EXEC_LOG_FOLDER  #"/nfs/archive"
    if type=="apk_zip":
        filename=os.path.join(ARC_DIR,rid,fname)
        rfilename=os.path.join(RLT_DIR,rid,fname)
        if os.path.exists(filename): # for archive dir
            return filename
        elif os.path.exists(rfilename): # for result dir?
            return rfilename
    elif type=="apk_log":
        wfname=os.path.join(ARC_DIR,rid, fname) #archive folder, fname="*.only.log.xposed"
        elist=glob.glob(wfname)
        if len(elist)>=1:   
            return elist[0]
        rfname=os.path.join(RLT_DIR,rid, fname) #result folder
        elist=glob.glob(rfname)
        if len(elist)>=1:   
            return elist[0]
    elif type=="__apk__": # APK for emulator
		# return APK based on cid >= int(rid)
        PRED_DIR=settings.EXEC_PRED_FOLDER  #"/nfs/prediction"
        afname=os.path.join(PRED_DIR,"*.apk")
        apk_list=sorted(glob.glob(afname), key=lambda s: get_id_from_str(os.path.basename(s)))
        #print apk_list
        if len(apk_list)>=1:  
            # get first fname greater than rid
            for i in apk_list:
                cid=get_id_from_str(os.path.basename(i))
                if cid >= int(rid) and cid !=sys.maxint:
                    return i
        return None
    return None

#============================================================= get_result_file==================
# for sorting by id
def get_id_from_str(str):
    ret= sys.maxint
    try:
        ret=int(str.split('-')[0])
    except:
        pass
    return ret
    
#============================================================= get_result_file==================
# generic function to get text file in result folder
def get_result_file(rid, fname, ln, perm,disabled4reader, ftype, sort_col=None):
    print 'in get_result_file(), rid=', rid, ",ln=", ln, ",ftype=", ftype,",sort_col=",sort_col
    # check access
    document =_list.get_doc(rid, perm)
    #print "document..=",document
    if not document:
        return Response({"error":"doc not found"},status=404)
        
    # get data from result folder
    rf=os.path.join(settings.RESULT_DIR_FULL,str(rid),fname)
    print "filename=",rf
    if not os.path.isfile(rf):
        ret={"id":rid, "fname":fname, "txt":"Warning: data not available!"} 
        return HttpResponse(json.dumps(ret), content_type="application/json")
    
    ml_opts={}
    if not document.ml_opts is None:
        ml_opts=json.loads(document.ml_opts)
        
    #print document.ml_opts #json.loads(document.ml_opts)
    if "has_excluded_feat" in ml_opts and ml_opts["has_excluded_feat"]==1:
        has_excluded_feat=1
    else:
        has_excluded_feat=0
    
    print "has_excluded_feat=", has_excluded_feat

    # return json file, for feature list
    if ftype == "json":
        try:
            with open(rf,'r') as jf:
                ret=json.load(jf)
        except : 
            return Response({"error":"open file error!"},status=404)
        # sort_col has sort order,to float, to absolute value flags and json field name for sorting  
        # assume array in key="data" (tbd)
        print "ret t=",type(ret)

        # for feat list of prediction, ret is a list
        if type(ret) is list:
            return HttpResponse(json.dumps(ret), content_type="application/json")
        
        # for feat list in train page. expect data in a dict["data"] for sorting and filtering
        if sort_col:
            arr=ret["data"]
            # sort order flag
            if sort_col[0:1]=='-':
                sort_col=sort_col[1:len(sort_col)]
                reverse=True
            else:
                reverse=False
            # convert to float flag
            if len(sort_col)>3 and sort_col[0:4]=='flo_':
                sort_col=sort_col[4:len(sort_col)]
                flo=True
            else:
                flo=False
            # abs flag
            if len(sort_col)>3 and sort_col[0:4]=='abs_':
                sort_col=sort_col[4:len(sort_col)]
                abs=True
            else:
                abs=False

            try:
                if flo and abs : # abs() somehow didn't work here...
                    sarr=sorted(arr, key=lambda k: float(k[sort_col]) if float(k[sort_col]) > 0.0 else float(k[sort_col])* -1, reverse=reverse)
                elif flo:
                    sarr=sorted(arr, key=lambda k: float(k[sort_col]), reverse=reverse)
                else:
                    sarr=sorted(arr, key=lambda k: k[sort_col], reverse=reverse)

                # get first "ln" lines    
                if ln > "0":
                    ret_ln=int(ln)
                    #print "ret_ln=",ret_ln,",len(sarr)=",len(sarr),",has_excluded_feat=",has_excluded_feat
                    if has_excluded_feat==1 and len(sarr) > ret_ln:
                        # if having excluded feature, need to include them
                        ret_arr=sarr[0:ret_ln]
                        # get excl list from mongo
                        ex_list=ml_util.ml_get_dataset_info(rid,"feature_excluded")
                        # convert to int
                        ex_list=[ int(i) for i in ex_list ]
                        # create dict
                        ex_dict=dict(zip(ex_list,ex_list))
                        #print "ex_list=",ex_list,",ex_dict=",ex_dict
                        # add excluded feat to return list
                        for idx, jo in enumerate(sarr[ret_ln:]):
                            if jo["coef"]==0 and int(jo["fid"]) in ex_dict:
                                ret_arr.append(jo)
                                #print "jo=",jo
                        ret["data"]=ret_arr
                    else:
                        ret["data"]=sarr[0:ret_ln]
                else:
                    ret["data"]=sarr
            except : 
                e = sys.exc_info()[0]
                print "Warning Sort error!! %s" %e
        elif ln > "0": # return only ln lines
            try:
                ret["data"]=ret["data"][0:int(ln)]
            except : 
                e = sys.exc_info()[0]
                print "Warning line extract error!! %s" %e
    elif ftype == "json_csv": # assume json has one level depth, no double quote in data
        try: # convert json to csv
            with open(rf,'r') as jf:
                ret=json.load(jf)
                if "data" in ret:
                    ret=ret["data"]
                csvtxt=",".join(ret[0].keys())+"\n"
                for row in ret:
                    csvtxt=csvtxt+",".join(map( lambda x: '"'+str(x)+'"', row.values()) )+"\n"
                return HttpResponse(csvtxt, content_type="application/text")
        except:
            return Response({"error":str(sys.exc_info()[0])},status=404)
    else: # return all TEXT content; 
        #txt="fid\tn-gram\t\traw_string\n"
        try:
            #  avoid complexity, read all logs
            for i, line in enumerate( sorted(fp.readlines()) ):
                #print i,",l=",line
                #if i >= startln:
                txt=txt+line
                #endln=i
            ret ={"id":rid, "fname":fname, "txt":txt} 
        except : 
            return Response({"error":"open file error!!!"},status=404)
            
    #print "ret data len=",len(ret["data"])
    return HttpResponse(json.dumps(ret), content_type="application/json")

#============================================================= api_post_apk==================
# submit APK or get APK state
def get_post_apk(request, cid, perm,disabled4reader):
    print "in get_post_apk()"
    return _emulator.emulate(request, rid=None, cid=cid, msg_id=None, perm=perm, disabled4reader=disabled4reader, from_api="y")

#============================================================= api_post_apk==================
# create dataset entry in db
def create_ds(request, perm,disabled4reader):
    action=request.POST.get('action')
    print "********* action=", action
    #print "request.POST.get('ptn_str')=",request.POST.get("ptn_str")
    #return Response({"error":"hihi"},status=404)
    if action is None or not action in ("hdfs_api","ensemble_api"): 
        return Response({"error":"not supported"},status=404)
    # check security
    rid= _list.list2(request,0,"",perm,disabled4reader)
    ret={"id":rid}
    return Response(ret)

    
#============================================================= extract_feature ==================
# featuring by api
def extract_feature(request, perm, disabled4reader):
    action=request.POST.get('action')
    rid=request.POST.get('hf_w_id')
    if action is None or action != "feature_api": 
        return Response({"error":"not supported"},status=404)
    if rid is None:
        return Response({"error":"id not found"},status=404)
    # check doc
    document =_list.get_ds_doc(rid, perm)
    #print "document..=",document
    if document is None:
        return Response({"error":"dataset not found"},status=404)
        
    rid, msg_id, ret_msg=_list.ml_opts(request,perm,disabled4reader)
    ret={"id":rid, "msg_id":msg_id, "ret_msg":ret_msg}
    return Response(ret)
    
#============================================================= train ==================
# train by api
def train(request, perm, disabled4reader):
    action=request.POST.get('action')
    rid=request.POST.get('hf_w_id')
    if action is None or not action in ("mllib_api","scikit_api"): 
        return Response({"error":"not supported."},status=404)
    if rid is None:
        return Response({"error":"id not found"},status=404)
    # check doc
    document =_list.get_ds_doc(rid, perm)
    #print "document..=",document
    if document is None:
        return Response({"error":"dataset not found"},status=404)
        
    rid, msg_id, ret_msg=_list.ml_opts(request,perm,disabled4reader)
    ret={"id":rid, "msg_id":msg_id, "ret_msg":ret_msg}
    return Response(ret)
    
#============================================================= set_data ==================
# set data for enasemble or for gpu worker
def set_data(request, type, rid, perm, disabled4reader):
    # check support types
    if not type in ("_es_list","dnn_state"):
        return Response({"status":"failed","msg":"not supported"},status=404)
    
    # check doc
    document =_list.get_ds_doc(rid, perm)
    #print "document..=",document
    if document is None:
        return Response({"status":"failed","msg":"record not found"},status=404)
    
    if "ensemble" in document.file_type:
        ds_list=request.POST.get("hf_w_ds_list")
        document.ds_list=ds_list
        document.save()
        return Response({"status":"updated","id": rid,"msg":"Dataset list updated for Id="+rid})
    elif  type in ("dnn_state"):
        dnn_state=request.POST.get("dnn_state")
        document.ml_state=dnn_state
        document.save()
        return Response({"status":"updated","id": rid,"msg":"succeeded"})
    else:
        return Response({"status":"failed","msg":"not an ensemble record"},status=404)
 
#============================================================= rm_data ==================
# del table row from web; TBD to clean related data
def rm_data(rid, type, perm, disabled4reader):
    # for deleting dataset record only 
    if type=="ds":
        document =_list.get_ds_doc(rid, perm)
    elif type=="pred":
        document =_predict.get_pred_doc(rid, perm,disabled4reader)
        if not document is None and len(document)>0:
            document=document[0]
        
    if document is None:
        return Response({"status":"failed","msg":"record not found"},status=404)
    # should we not really delete record?
    ret=document.delete()
    print "ret=", ret
    if ret[0]==1:
        return Response({"status":"deleted","msg":"Record id="+rid+" deleted"})
    else:
        return Response({"status":"failed","msg":"Delete failed for id="+rid},status=404)
 
#============================================================= query action ==================
# do action without changing Db
def api_query(request, type, rid, perm, disabled4reader):
    # check support types
    if not type in ("verify_dnn_model"):
        return Response({"status":"failed","msg":"type not supported"},status=404)
    rid=request.POST.get('hf_w_id')
    learning_algorithm=request.POST.get("learning_algorithm")
    ml_model=request.POST.get("hf_w_ml_model")
    ml_opts=request.POST.get("hf_w_ml_opts")
    ret=verify_dnn_model(ml_model,ml_opts)
    return Response(ret)
    
#============================================================= verify DNN model ==================
# 
def verify_dnn_model(jstr_model, ml_opts, keras_lib_dir=config.get('env', 'KERAS_LIB_DIR')):
    import sys
    sys.path.append(keras_lib_dir)
    from keras.models import model_from_json
    
    #print "ml_opts=",ml_opts
    jopts=json.loads(ml_opts)
    try:
        model = model_from_json(jstr_model)
        #model = model_from_json('{"class_name": "xxx","config":[]}')
        model.compile( loss=jopts['loss'], optimizer=jopts['optimizer'] )
        # remove special chars
        c_model=jstr_model.replace("\n","").replace("\r","")
        #print "jstr_model=",jstr_model
        # get the output from print(model.summary())
        cmd="import sys;sys.path.append('"+keras_lib_dir+"');from keras.models import model_from_json;" \
            +"model=model_from_json('"+c_model+"');" \
            +"model.compile(loss='"+jopts['loss']+"',optimizer='"+jopts['optimizer']+"');" \
            +"print(model.summary());" \
            +"print sys.exc_info()[0] "
        proc = subprocess.Popen(["python", "-c", cmd], stdout=subprocess.PIPE)
        output="Input model:\n"+jstr_model \
            +"\nLoss='"+jopts['loss']+"', optimizer='"+jopts['optimizer']+"'" \
            +"\nNetwork Summary:\n"+proc.communicate()[0]
        return {"status":"succeeded","msg":"Model compile successfully!","output":output}
    except Exception as e:
        output="Model:\n"+jstr_model+"\nResult:\n"+e.__doc__+"\n"+e.message
        return {"status":"failed","msg":"Model compile failed!!","output":output}


    