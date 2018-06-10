# -*- coding: utf-8 -*-
'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''
from django.conf.urls import  include, url
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView
#from atdml import views

from django.contrib import admin
from django.contrib.auth.views import *
from atdml.views import *
admin.autodiscover()

urlpatterns = [#'',
	url(r'^atdml/', include('atdml.urls') ),
	url(r'^$', RedirectView.as_view(url='/atdml/list/') ), 
    
] 

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


urlpatterns += [#'',
    # tbd. user home page?
	#url(r'^user/', include('user_acct.urls'), name='user'),

    #admin page
    url(r'^admin/', include(admin.site.urls)),

    # for login
    url(r'^atdml/', list , name='atdml_list'),
    url(r'^about_mlaas/$', about_mlaas , name='about_mlaas'),
    url(r'^about_ae/$', about_ae , name='about_ae'),
    url(r'^about_ae$', about_ae , name='about_ae'),
    url(r'^help_mlaas/$', help_mlaas , name='help_mlaas'),
]

# for login logout'django.contrib.auth.views',
urlpatterns += [
    #'django.contrib.auth.views',

    
    url(r'^login/$', login_recaptcha, {'template_name': 'login.html'}, name="myml_login"),
    url(r'^login/(?P<msg_id>[0-9.]+)/$', login_msg, {'template_name': 'login.html'}, name="login_msg"),

	#url(r'^login/$', login, {'template_name': 'login.html'}, name="myml_login"),

	url(r'^logout/$', logout, {'next_page': 'myml_login'}, name="myml_logout"), # point back to login page
        
	url(r'^logout$', logout, {'next_page': 'myml_login'}, name="myml_logout"), # point back to login page
        

    url(r'.*', list , name='default'),

	
] 




