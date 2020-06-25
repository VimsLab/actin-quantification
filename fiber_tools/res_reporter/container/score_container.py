from .container import Container
import numpy as np
import pandas as pd

import pdb
class ScoreContainer(Container):

    def __init__(self, filelist):

        Container.__init__(self, filelist)


    def calc_sigmoid(self):
        self.df['sigmoid'] = self.df['out'].apply(sigmoid)


def sigmoid(feature):
    return 1 / (np.exp(-np.array(feature, dtype=np.float32)) + 1)