/****************************************************************************
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
****************************************************************************/
	// variables for Log Popup
    var log_timer;					// timer for log polling
    var log_timeout;				// timeout for log polling
	var WEBROOT='/atdml';			// for HTTP GET
	var logtype_id='#_h_logtype'  	//for "View Log" button
	var logrid_id='#_h_rid'   		//for "View Log" button
	var logpro_img='#_prg_img_log'	// id for prog circle
	var logtextbody='#_logtextbody'	// log text area
	var logshow_btn='._showlog_btn'	// view log button
	var loglabel_id='#_logModalLabel'	// log popup label
	var logtimer_id='#_log_timer'		// log timer
    
    //train opt id, TBD to pass in?
    var hf_w_oid='#hf_w_oid'
	
    // timer for for msg_info
    var info_timer;
    var info_timeout;
	var msg_info_timer_id='#_msg_info_timer'	// time in msg box
	var msg_success_timer_id='#_msg_success_timer'	// time in msg box
	
    // enable or disable a HTTP object ====================================
    function setDisable(objId, torf){       
        //$(objId).css({'display','inline'}); 
        //alert(hors)
        // somehow it works reverse way for file upload...?
        
        if( torf == "false") {
            $(objId).prop("disabled",false);
        } else{
            //alert("in show="+hors);
            $(objId).prop("disabled",true);
        } 
    };

    // hide or show a HTTP object  ====================================
    function setHideShow(objId, hors){       
        //$(objId).css({'display','inline'}); 
        //alert(hors+" hihi????")
        //$(objId).toggle(); didn't work
        // somehow it works reverse way...?
        if( hors == "hide") {
            $(objId).hide();
            //stop_timer(msg_info_timer_id, msg_success_timer_id);
        } else{
            //alert("in show="+hors);
            $(objId).css({"display":"inline"});
            $(objId).show();
        } 
    };

    // hide all message panels ====================================
    function hideAllMsgPan(){
        $("div._msgpan_").hide();
        stop_timer(msg_info_timer_id);
    }  

    // set text to message panel  ====================================
    function setMsgPanText(msg,type,hideMsgMsec,rid=""){
        hideAllMsgPan();
        var pan= $("#msg_"+type); 
        var pan_cnt=$("#msg_"+type+"_ph"); 
 
        // replace tag with id if _xidx_ exists
        if (rid.length>0)
            pan_cnt.html(msg.replace("_xidx_",rid));
        else
            pan_cnt.html(msg);
//alert("pan_cnt="+pan_cnt.text());
        pan.css({"display":"inline"});
		// ????
        //start_timer(msg_info_timer_id);
        pan.show();
		// to close msg pan automatically
        if (hideMsgMsec>0) {
            setTimeout(function(){
                    //alert("hide?");
                    setHideShow("#msg_"+type,"hide");
					stop_timer(msg_info_timer_id);
                }, hideMsgMsec
            );  
        }  
    }

    // action pane
    //  query and echo return
    function show_query(rid, url, form, title=null
        , imgId="#_prg_img_qry", tgt_txtbody="#_qrytextbody", modal_title="#_qryModalTitle"){

//alert("in show_act:"+url+title);
        $(modal_title).text(title);
		setHideShow($("#_prg_img_qry") ,"show");
        $(tgt_txtbody).val('Retriving data... ');	
        
        // submit to URL
        $.ajax({
          type: "POST",
          url: url,
          data: form.serialize()
        }).success(function(data){
            setHideShow(imgId,"hide");
        }).done(function(data){
//alert("done: "+data.opts+";"+data.status);
            $(tgt_txtbody).val(data.msg+"\n"+data.output)
        }).fail(function(jqXHR, textStatus){
            data=$.parseJSON(jqXHR.responseText);
//alert("fail textStatus:"+textStatus+";msg="+msg);
            $(tgt_txtbody).val(data.msg+"\n"+data.output)
            setHideShow(imgId,"hide");

        });
        
    }

	function stop_act(rid){
		//alert("in stop_act");
		setHideShow($("#_prg_img_qry") ,"hide");
	}
    

    // timer for polling job log at base.html
    //  call query_set_log()
    function show_log(rid, logtype){
		var linenumb=0;
		//var logtxt="";
		//var url=WEBROOT+'/log/'+rid+'/'+logtype+'/'+linenumb+'/'
        //alert("hello id="+rid+",t="+logtype+'\nurl='+url);
        $(logtextbody).val('Retrieving log ...');	
		setHideShow(logpro_img ,"show");
		log_timer=setInterval( function(){
                query_set_log(rid, logtype, linenumb,logtextbody);
                //alert("sts: " + data.status + "\nLog: " + data.log);
            }, 2900);
            /*function(){
            $.get(url, 	function(data,status){
					logtxt=data.log;
					if (logtxt>""){
						$(logtextbody).val(logtxt);		//plain text			
						$(logtextbody).scrollTop($(logtextbody)[0].scrollHeight); // scroll to bottom
					}
					//alert("sts: " + data.status + "\nLog: " + data.log);
					if(data.status.indexOf("processing")==-1){
						stop_log(rid);
					}
				});
        }, 2900);*/
        info_timeout=setTimeout(function(){
            clearInterval(log_timer);
        }, 3600000); // for 1 hour
    }
	// log file located at URL: WEBROOT/<id>/<logtype>/<offset>
	// use HTTP GET to pull log info
    //   [logtype] used to find log file by name ${rid}logtype or {pred_id}predict
    function query_set_log(rid, logtype, offset, tgt_txtbody){
        // not set prog image, not disable button, no init msg
        query_set_log_prog(rid, logtype, offset, tgt_txtbody, null, null, null)
    }
    function query_set_log_prog(rid, logtype, offset, tgt_txtbody ,imgId, btnId, initMsg){
        if (Number(rid) <=0) return;
		var logtxt="";
        if(logtype=="")
            logtype=$(logtype_id).val();
        if(rid==0)
            rid=$(logrid_id).val();
		var url=WEBROOT+'/log/'+rid+'/'+logtype+'/'+offset+'/'
        if(tgt_txtbody==""){
            tgt_txtbody=logtextbody;
            //alert("query_log id="+rid+",t="+logtype+'\nurl='+url);
        }
        // set info pane
        if (initMsg>"")
            setMsgPanText(initMsg, "info",0);
        // turn on progress image
        if (imgId>"")
            setHideShow(imgId,"show");
        // disable action button for record rid
        if (btnId>"")
            setDisable(btnId,"true");
        
        $.get(url, 	function(data,status){
            var logtxt=data.log;
            var fsize=data.fsize;
//alert("fsize="+fsize+", logtxt s="+logtxt.length); 
            //if (fsize>logtxt.length)
            //    logtxt=logtxt+'\n...... Log size is huge. Only last part was shown. Click "Refresh Log" to load full content.'    
            if (logtxt>""){
                $(tgt_txtbody).val(logtxt);		//plain text	
                if ( $(tgt_txtbody)[0] != null)
                    $(tgt_txtbody).scrollTop($(tgt_txtbody)[0].scrollHeight); // scroll to bottom
            }
            //alert("before stop, logrid="+$(logrid_id).val()+", sts="+data.status);
            // stop log timer if rid is current and status is not processing
            if(rid==$(logrid_id).val() && data.status.indexOf("processing")==-1){
                //alert("in stop");
                stop_log(rid);
            }
        }).done(function(){
            // set success info pane
            if (initMsg>"")
                setMsgPanText("Log retrieval for ["+logtype+"] completed at "+getDateTime(), "success",0);
            // turn off progress image
            if (imgId>"")
                setHideShow(imgId,"hide");
            // enable action button 
            if (btnId>"")
                setDisable(btnId,"false");
        });
    }
    
	// stop polling and hide progress/running circle image
	function stop_log(rid){
		//alert("in stop_log");
        clearInterval(log_timer);
        clearTimeout(log_timeout);
		setHideShow(logpro_img ,"hide");
        if (rid >""){
            $(loglabel_id).text("Excution log for Id: "+rid);
        }
	}
    // get file from result folder; tbd make it more generic?
    function get_set_file(rid, fname, linenumb, tgt_txtbody ,imgId, btnId, initMsg){
        if (Number(rid) <=0) return;
		var txt="";

		var url=WEBROOT+'/api/f/'+rid+'/'+fname+'/'+linenumb+'/'

        // set info pane
        if (initMsg>"")
            setMsgPanText(initMsg, "info",0);
        // turn on progress image
        if (imgId>"")
            setHideShow(imgId,"show");
        // disable action button for record rid
        if (btnId>"")
            setDisable(btnId,"true");
        
        $.get(url, 	function(data,status){
            txt=data.txt;
            if (txt>""){
                $(tgt_txtbody).val(txt);		//plain text	
                if ( $(tgt_txtbody)[0] != null)
                    $(tgt_txtbody).scrollTop($(tgt_txtbody)[0].scrollHeight); // scroll to bottom
            }
            //alert("before stop, logrid="+$(logrid_id).val()+", sts="+data.status);
            //alert("data="+data+", status="+status);
            console.log(data);
            // stop log timer if rid is current and status is not processing
            if(rid==$(logrid_id).val() && data.status.indexOf("processing")==-1){
                //alert("in stop");
                stop_log(rid);
            }
        }).fail(function(jqXHR, textStatus){
            //alert("error at get.fail().");
        }).done(function(){
            // set success info pane
            if (initMsg>"")
                setMsgPanText("Data retrieval for ["+fname+"] completed at "+getDateTime(), "success",0);
            // turn off progress image
            if (imgId>"")
                setHideShow(imgId,"hide");
            // enable action button 
            if (btnId>"")
                setDisable(btnId,"false");
        });
    }
    
	// stop polling and hide progress/running circle image
	function stop_log(rid){
		//alert("in stop_log");
        clearInterval(log_timer);
        clearTimeout(log_timeout);
		setHideShow(logpro_img ,"hide");
        if (rid >""){
            $(loglabel_id).text("Excution log for Id: "+rid);
        }
	}
    
    
	// set log type for display
	function set_logtype(logtype,rid,initMsg){
        if (logtype>""){
            $(logtype_id).val(logtype);
            var oid=$(hf_w_oid).val()
            // set to oid for training option record
            if (oid !=null && oid != rid){
                $(logrid_id).val(oid);
                rid=oid;
            } else{
                $(logrid_id).val(rid);
            }
            //alert("hf_w_oid="+oid);
            //$(logrid_id).text(rid);
            //alert("initMsg="+initMsg);          
			$(loglabel_id).text(initMsg);
			setHideShow(logpro_img ,"show");
			setHideShow(logshow_btn ,"show");
        } else {
            $(logtype_id).val('');
            $(logrid_id).val('');
            //$(logrid_id).text('');
			$(loglabel_id).text("Excution log for Id: "+rid);
			setHideShow(logpro_img ,"hide");
			setHideShow(logshow_btn ,"hide");
        }
	}

    /* submit a form
        formId: id of a html form
        imgId: id of progress image to show and then hide after done
        btnId: id/class of action button to disable and then enable after done
        rid: record id
        initMsg: initial info message after button click
        refreshCallback: function to refresh UI after ajax
        logtype: log file type
    */
    function submit_job(formId, imgId, btnId, rid, initMsg, refreshCallback, logtype){
        return submit_job2(formId, imgId, btnId, rid, initMsg, refreshCallback, logtype
            , null, null, null ,"1" ,"1")
    }
    function submit_job2(formId, imgId, btnId, rid, initMsg, refreshCallback, logtype
                , stype, surl, sdata, done_msg_flag, success_msg_flag){
        // avoid default action
        //event.preventDefault();
        // set params for log retrieval
        set_logtype(logtype,rid,initMsg);

        // set initial info message
        if (initMsg>"")
            setMsgPanText(initMsg, "info",0);
        else
            setMsgPanText("Processing Id="+rid+"... ", "info",0);
		// start the clock
		start_timer(msg_info_timer_id);
        // turn on progress image
        setHideShow(imgId,"show");
        // disable acton button for record rid
        setDisable(btnId,"true");

        // get form
        var frm = $( formId );
        if (stype == null) 
            stype=frm.attr("method");
        if (surl==null)   
            surl=frm.attr("action");
        if (sdata==null)
            sdata=frm.serialize();
        
        //alert(formId+","+imgId+","+frm.attr("method")+","+frm.attr("action")+","+frm.serialize());
        //alert("stype="+stype+", surl="+surl+", sdata="+sdata);
        
        // submit Ajax here =======================
        $.ajax({
            type: stype, // frm.attr("method"),
            url: surl, //frm.attr("action"),
            timeout: 0, //Set your timeout value in milliseconds or 0 for unlimited
            data: sdata //frm.serialize()
            //,dataType: "json"
        }).success(function(data){
//alert("success: msg="+data.msg);
            var msg=data.msg
            if (success_msg_flag=="1"){
                if (msg>"")
                    setMsgPanText(msg, "success", 0, rid);            
                else
                    setMsgPanText("Done Id="+rid+"!", "success", 0);
            }
            // invoke callback for refresh UI
            refreshCallback(data);

        }).done(function(data){
//alert("done: ");
            if (done_msg_flag=="1"){
                setHideShow(imgId,"hide");
                stop_timer(msg_info_timer_id, msg_success_timer_id);
                setDisable(btnId,"false");
            }
        }).fail(function(jqXHR, textStatus){ 
        //}).fail(function(data){
//alert("fail textStatus:"+textStatus);
//console.log(textStatus);
//console.log(jqXHR);
//alert("rt="+jqXHR.responseText);
            var http_code=null;
            var statusText=null;
            var msg=null;
            var data=null;
            if (jqXHR != null ){
                try {
                    http_code=jqXHR.status;
                    statusText=jqXHR.statusText
                    if (http_code==503){
                        msg=statusText+". Ajax request may be reset by proxy. Please check job progress in the Log page."
                    }
                    data=$.parseJSON(jqXHR.responseText);
                } catch (e) {  
//alert("exception at data=$.parseJSON");
                }
            }
            if (data != null){
                msg=data.msg
            }
            //alert("msg="+msg)
            if (msg!=null && msg.length>0){
                setMsgPanText(msg, "error", 0);            
            }else{
                setMsgPanText("Error on Id="+rid+"!", "error", 0);
            }
            setHideShow(imgId,"hide");
			stop_timer(msg_info_timer_id, msg_success_timer_id);
            setDisable(btnId,"false");
        });
    } // end submit_job



    // stop timer for msg_info
    function stop_timer(txtId, successId){
        clearInterval(info_timer);
        clearTimeout(info_timeout);
		//post elapsed time to success msg box
		if (successId>"" && $(txtId).text()>"")
			$(successId).text($(txtId).text())
		stop_log("")
        query_set_log(0, "", 0,logtextbody)
        // keep the time $(txtId).text("");
    }
    
    // start timer for msg_info
    function start_timer(txtId){
		//alert("in start_timer() 2");
        var earlierDate=new Date();
        var t="";
        info_timer=setInterval(function(){
            t=get_time_diff_str(earlierDate);
            $(txtId).text(" ..Elapsed time="+t);
			$(logtimer_id).text("Elapsed time="+t);
        }, 1000); // timer click every sec
        info_timeout=setTimeout(function(){
            clearInterval(info_timer);
            $(txtId).text("... Task taking more than 10hr!");
			$(logtimer_id).text("Elapsed time more than 10hr!");
        }, 36000000); // for 10 hour
        //return timer;
    }

    // return time diff string
    function get_time_diff_str(earlierDate) 
    {
        var oDiff = new Object();
        //earlierDate=new Date('2014','12','3');
        var laterDate=new Date();
        
        //  Calculate Differences
        //  -------------------------------------------------------------------  //
        var nTotalDiff = laterDate.getTime() - earlierDate.getTime();

        oDiff.days = Math.floor(nTotalDiff / 1000 / 60 / 60 / 24);
        nTotalDiff -= oDiff.days * 1000 * 60 * 60 * 24;

        oDiff.hours = Math.floor(nTotalDiff / 1000 / 60 / 60);
        nTotalDiff -= oDiff.hours * 1000 * 60 * 60;

        oDiff.minutes = Math.floor(nTotalDiff / 1000 / 60);
        nTotalDiff -= oDiff.minutes * 1000 * 60;

        oDiff.seconds = Math.floor(nTotalDiff / 1000);

        //  Format Duration
        //  Format Hours
        var hourtext = '00';
        if (oDiff.days > 0){ hourtext = String(oDiff.days);}
        if (hourtext.length == 1){hourtext = '0' + hourtext};

        //  Format Minutes
        var mintext = '00';
        if (oDiff.minutes > 0){ mintext = String(oDiff.minutes);}
        if (mintext.length == 1) { mintext = '0' + mintext };

        //  Format Seconds
        var sectext = '00';
        if (oDiff.seconds > 0) { sectext = String(oDiff.seconds); }
        if (sectext.length == 1) { sectext = '0' + sectext };

        //  Set Duration
        var sDuration = hourtext + ':' + mintext + ':' + sectext;
        oDiff.duration = sDuration;

        //return oDiff;
        return sDuration
    };
    // get datetime
    function getDateTime() {
        var now     = new Date(); 
        var year    = now.getFullYear();
        var month   = now.getMonth()+1; 
        var day     = now.getDate();
        var hour    = now.getHours();
        var minute  = now.getMinutes();
        var second  = now.getSeconds(); 
        if(month.toString().length == 1) {
            var month = '0'+month;
        }
        if(day.toString().length == 1) {
            var day = '0'+day;
        }   
        if(hour.toString().length == 1) {
            var hour = '0'+hour;
        }
        if(minute.toString().length == 1) {
            var minute = '0'+minute;
        }
        if(second.toString().length == 1) {
            var second = '0'+second;
        }   
        var dateTime = year+'-'+month+'-'+day+' '+hour+':'+minute+':'+second;   
        return dateTime;
    }

    // general input validation function   ====================================
    //TBD for val_func
    function validate_highlight(val_func, cntr_id, hgh_id, err_msg){
        //alert("in validate_highlight:" +$(cntr_id).val());
        var ret=false;
        
        if (val_func=="" || val_func==null){
            // default for empty
            if ($(cntr_id).val()==""){
                setMsgPanText(err_msg, "error",0);
                $(hgh_id).addClass('has-error');
                return ret;
            } else {
                $(hgh_id).removeClass('has-error');
            }
        }
        ret=true;
        return ret;
        //TBD for val_func
    }
    
    // Input validation, support: int, float, greater than/equal 0, less than 1 ##########################
    // required attribute "validate_rule" and element <element id>_error_help_msg to show error 
    // based on Bootstrap class "has-error", "input-group"  to highlight error element
    //          local class "mlhide" to hide/show error message
    function validate_by_rule(element,msg_element){
        var str_input=element.val().trim();
        var str_rule=element.attr('validate_rule');
        var err=0
        //alert("str_rule="+str_rule+",str_input="+str_input);
        // check int
        if (err==0 && str_rule.indexOf('int')>=0 &&  str_input != parseInt(str_input, 10)) err=1
        // check float
        //if (err==0 && str_rule.indexOf('float')>=0 &&  /^-?\d+$/.test(str_input) ) err=1
        if (err==0 && str_rule.indexOf('float')>=0 &&  str_input!=parseFloat(str_input))  err=1
        //alert("parseF="+parseFloat(str_input))
        //alert("isNaN="+isNaN(parseFloat(str_input)))
        // check gt0
        if (err==0 && str_rule.indexOf('gt0')>=0 &&  str_input <= 0 ) err=1
        // check ge0
        if (err==0 && str_rule.indexOf('ge0')>0 &&  str_input < 0 ) err=1
        // check lt1
        if (err==0 && str_rule.indexOf('lt1')>=0 &&  parseFloat(str_input) >= 1 ) err=1
        // check non empty string
        if (err==0 && str_rule.indexOf('nempt')>=0 &&  str_input !=null &&  str_input.length == 0 ) err=1
        // check int list
        if (err==0 && str_rule.indexOf('list')>=0 &&  str_input !=null &&  str_input.length > 0 ) {
            var new_list=str_input.split(',');    
            for (var i=0;i<new_list.length; i++){
                if (new_list[i] != parseInt(new_list[i], 10)) {                 
                    err=1;
                    break;
                }    
            }
        }
        // check json
        if (err==0 && str_rule.indexOf('json')>=0){
            
            try {
                j=JSON.parse(str_input);
                //console.log(j);
            } catch (e) {
                err=1
            }
        }

        if (err==1){
            //alert("error");
            element.closest('.input-group').addClass('has-error'); 
            //show error message
            msg_element.removeClass('mlhide');
            return false;
        } else {
            element.closest('.input-group').removeClass('has-error'); 
            //msg_element.css({"display":"none" }); 
            //hide error message
            msg_element.addClass("mlhide");
            return true;
        }
    }            
    