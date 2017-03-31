import numpy as np
from model import *
import numericalAnalysis
import matplotlib.pyplot as plt
import time
import sys
import pandas
import utils

resultsDir = "C:/Users/pkhallouf/Documents/PK/RM-MODEL/results/"

initialState = np.array([1, 1, 1, 1])
parameters = {
"a": 0.3,
"b": 0.3,
"e": 0.5,
"m": 0.4,
"d": 0.5
}

timeStep = 5e-2
horizon = 5e4
numberTimeStep = int(horizon / timeStep)

#isolatedSystem = Rosenzweig_MacArthur(initialState[0:2], parameters)
coupledSystem = Coupled_RM(initialState, parameters)

#results_old, a_sample_old = readPermanentStatesFromExcel("C:/Users/pkhallouf/Documents/PK/RM-MODEL/results/results__b20_e50_d50_m40.xlsx", popName)

b_sample = np.round(np.arange(0.05, 0.43, 0.01),2)
a_sample = np.round(np.arange(0.45, 10, 0.1), 2)

#numericalAnalysis.bifurcationCartography(coupledSystem, a_sample, b_sample, resultsDir, timeStep, horizon, checkExisting="_da005")
#numericalAnalysis.bifurcationCartography(coupledSystem, a_sample, b_sample, resultsDir, timeStep, horizon, checkExisting=numericalAnalysis.SKIP_IF_EXISTING)

bifurcationMap = np.empty((a_sample.size, b_sample.size, coupledSystem.dim))

for bi, b in enumerate(b_sample):
    print("getting results for b = %f" %b)
    coupledSystem.update(parameters={"b":b})
    fileName = resultsDir + "results__b" + str(int(b * 100)).zfill(2) + "_e50_d50_m40.xlsx"
    figName = resultsDir + "/figures/extrema"
    for p in sorted(coupledSystem.parameters.keys(), key = lambda x: coupledSystem.paramOrder[x]):
        if p == "a": continue
        figName += "_" + p + str(round(coupledSystem.parameters[p], 2))
    figName += ".png"

    res, a_array = numericalAnalysis.readPermanentStatesFromExcel(fileName, coupledSystem.popName, a_sample)

    fig = utils.plotBifurcation(coupledSystem, res[:,0:3,:], a_array, saveFileName=resultsDir + figName)
    fig.savefig(figName)

#    if (a_array.size != a_sample.size) or (abs(a_array - a_sample).max() > 1e-8).any():
#        sys.exit("a samples do not match !")
#    amplitude = res[:,0,:] - res[:,1,:]
#    bifurcationMap[:, bi, :] = amplitude


#x,y = np.meshgrid(a_sample, b_sample)
#
#fig = plt.figure(1)
#nbPlots = coupledSystem.dim
#nbRows = len(coupledSystem.subSystems)
#nbCols = coupledSystem.subSystems[0].dim
#for i in range(nbRows):
#    for j in range(nbCols):
#        pop = len(coupledSystem.subSystems) * i + j
#        values = sorted(list(set(np.round(bifurcationMap[:,:,pop].reshape(bifurcationMap[:,:,pop].size),2))))
#        plt.subplot(nbRows, nbCols, pop+1)
#        plt.contourf(x,y,np.transpose(bifurcationMap[:,:,pop]), levels=values)
#        plt.colorbar()
#        plt.title("population %s" % coupledSystem.popName[pop])
#        plt.xlabel('parameter a')
#        plt.ylabel('parameter b')
#fig.tight_layout()
#fig.show()
#fig.savefig(resultsDir+"bifurcationMap_e05_d05_m04.png")