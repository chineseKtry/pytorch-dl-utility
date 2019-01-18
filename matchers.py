from __future__ import print_function, absolute_import
from fnmatch import fnmatch


def apply_matchers(named_modules, matchers, loss_t=None):
    for module_name, module in named_modules:
        for matcher in matchers:
            x = matcher.apply(module_name, module)
            if loss_t is not None and x is not None:
                loss_t += x
    return loss_t


class ModuleMatcher(object):
    def __init__(self, type_match=None, name_match=None, **args):
        self.type_match = type_match
        self.name_match = name_match
        self.args = args
    
    def match_name(self, module_name):
        return self.name_match is not None and fnmatch(module_name, self.name_match)
    
    def match_type(self, module):
        return self.type_match is not None and isinstance(module, self.type_match)
    
    def match(self, module_name, module):
        return self.match_name(module_name) or self.match_type(module)
    
    def apply(self, module_name, module):
        if self.match(module_name, module):
            return self.function(module)

    def function(self, module):
        raise NotImplementedError()


class WeightMatcher(ModuleMatcher):
    def match(self, module_name, module):
        return super(WeightMatcher, self).match(module_name, module) \
           and hasattr(module, 'weight') \
           and module.weight is not None
