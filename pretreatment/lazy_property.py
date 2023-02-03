# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:17:28 2015

Stolen from StackOverflow: 
http://stackoverflow.com/questions/3012421/python-lazy-property-decorator
"""

class lazy_property(object):
    '''
    meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    
    Usage:
    
    class Test(object):
        @lazy_property
        def results(self):
            calcs = # do a lot of calculation here
            return calcs
    '''

    def __init__(self,fget):
        self.fget = fget
        self.func_name = fget.__name__


    def __get__(self,obj,cls):
        if obj is None:
            return None
        value = self.fget(obj)
        setattr(obj,self.func_name,value)
        return value


