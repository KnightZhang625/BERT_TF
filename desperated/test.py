# coding:utf-8
# Produced by Andysin Zhang
# 06_Aug_2019

from log import log_error as _error

# Forbid to add new attributes function
#########################################################################
def forbid_new_attributes(wrapped_setatrr):
    def __setattr__(self, name, value):
        if hasattr(self, name):
            wrapped_setatrr(self, name, value)
        else:
            _error('Add new {} is forbidden'.format(name))
            raise AttributeError
    return __setattr__

class NoNewAttrs(object):
    __setattr__ = forbid_new_attributes(object.__setattr__)
    class __metaclass__(type):
        __setattr__ = forbid_new_attributes(type.__setattr__)
#########################################################################

if __name__ == '__main__':

    class A(NoNewAttrs):
        name = ''
        age = 0
        def __init__(self, a, b):
            self.name = a
            self.age = b

    a = A('abc', 20)
    print(a.name)
    print(a.age)
    a.name = 'k'
    print(a.name)
    a.address = 'a'