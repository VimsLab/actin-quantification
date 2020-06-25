import pandas as pd
import numpy as np

import pdb

class Container:
    def __init__(self, filelist):

        self.df = pd.DataFrame({'file':filelist})

    def set(self, name, value):
        if not isinstance(value, list):
            value = list(value)

        if len(value) == self.df.shape[0]:
            self.df[name] = value
        else:
            raise ValueError('On setting name: {}, len(value) {}, df shape {} -- shape mismatch'.format(name, len(value), self.df.shape))

    def get(self, name):
        return np.array(self.df[name].values.tolist())

    def get_raw(self, name):
        return self.df[name].tolist()

    def map(self, srcname, detname, func):
        self.df[detname] = self.df[srcname].apply(func)

