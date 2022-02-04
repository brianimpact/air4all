import json
import os


class Configuration:
    def __init__(self, dictionary):
        for k in dictionary.keys():
            if isinstance(dictionary[k], dict):
                dictionary[k] = Configuration(dictionary[k])
        self.__dict__ = dictionary
        
    def __getitem__(self, key):
        return self.dictionary[key]
    
    def __contains__(self, key):
        return key in self.dictionary.keys()