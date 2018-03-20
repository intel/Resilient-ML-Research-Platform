'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''

from django import template
from django.conf import settings

register = template.Library()

#https://docs.djangoproject.com/en/dev/ref/templates/builtins/?from=olddocs#linebreaksbr
    

# get values from settings =======================   
@register.simple_tag
def get_settings_value(name):
    #{% get_settings_value "MSG_UPLOAD_FAILED" %}
    return getattr(settings, name, "")    
    
# show partial string. not used yet... =============================
@register.filter
def show_partial(value):
    #{% variable|show_partial %}
    l=len(value)
    if l> 40:
        return value[:20]+"..."+value[l-10:l-1]
    return value    

# split by delimiter =============================
@register.filter
def split(value, delimiter):
    return value.split(delimiter)
    
@register.simple_tag
def split0(value, delimiter):
    return value.split(delimiter)[0]


    
    