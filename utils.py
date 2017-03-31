# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 15:18:01 2016

@author: pkhallouf
"""

import matplotlib.pyplot as plt

def plotBifurcation(system, table, xSample, saveFileName=None):
    fig = plt.figure()
    nbRows = len(system.subSystems)
    nbCols = system.subSystems[0].dim
    for i in range(nbRows):
        for j in range(nbCols):
            pop = len(system.subSystems) * i + j
            plt.subplot(nbRows, nbCols, pop+1)
            plt.plot(xSample, table[:,0,pop], label='max')
            plt.plot(xSample, table[:,1,pop], label='min')
            plt.plot(xSample, table[:,2,pop], label='mean')
            if system.popName[pop] in ["x", "y"]:
                plt.plot(xSample, 1.0/xSample, '--', label='carrying\ncapacity')
            elif system.popName[pop] in ["u", "v"]:
                plt.axhline(y=1./system.parameters["b"], linestyle='--', label='carrying\ncapacity')
#            plt.legend()
#            plt.title("population %s" % system.popName[pop])
            plt.xlabel('a')
            plt.ylabel(system.popName[pop])
    fig.tight_layout()
    return fig