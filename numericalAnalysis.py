# -*- coding: utf-8 -*-
import time
import numpy as np
import scipy
import pandas
import os

EPSILON = np.finfo(float).eps * 1e8 # ~= 2.22e-8
SKIP_IF_EXISTING = 'skip if existing'
USE_PREVIOUS_RESULTS = 'use previous results'

def nextStep(initialState, dynamics, timeStep, method="RK4"):
    dim = len(initialState)
    newState = np.empty(dim)
    if method == 'euler':
        newState = initialState + dynamics(initialState) * timeStep
    elif method == 'RK4':
        moveEstimate1 = dynamics(initialState)
        moveEstimate2 = dynamics(initialState + moveEstimate1 * timeStep * 0.5)
        moveEstimate3 = dynamics(initialState + moveEstimate2 * timeStep * 0.5)
        moveEstimate4 = dynamics(initialState + moveEstimate3 * timeStep)
        moveEstimate = (moveEstimate1 + 2 * moveEstimate2 + 2 * moveEstimate3 + moveEstimate4)
        newState = initialState + moveEstimate * timeStep / 6
#    else:
#        raise ValueError("Numerical resolution method %s could not be found" % method)
#    if detect_0_crossingEvent & (newState.min() < 0):
#        newState_corrected = initialState
#        newTimeStep = timeStep / 10
#        nbNewIter = 10
#        print(initialState, newState, timeStep, newTimeStep, nbNewIter)
#        for i in range(nbNewIter):
#            newState_corrected = nextStep(newState_corrected, dynamics, newTimeStep, method)
#        return newState_corrected
    return newState


def explicitSolver(dynamics, initialState, numberTimeStep, timeStep, method='RK4'):
    """
    F(X1) = F(X0) + dF(X0) * (X1 - X0)
    """
    dim = len(initialState)
    trajectory = np.zeros((numberTimeStep,dim))
    trajectory[0] = initialState
    for t in xrange(1, numberTimeStep):
        trajectory[t] = nextStep(trajectory[t-1], dynamics, timeStep, method)
    return trajectory

def getPermanentExtrema(system, timeStep=1e-5, convergenceCriteria=15, tolerence=1e-5, maxIteration=1e6):
    counterMax, counterMin = np.zeros((2, system.dim))
    localMax, localMin = (-10) * np.ones((2, system.dim))

    state_t0 = system.currentState
    state_t1 = nextStep(state_t0, system.systemDynamics, timeStep, method="RK4")

    nbIteration = 0
    while nbIteration < maxIteration:

        # Boost convergence by applying a flow
        if (nbIteration % 1e5 == 0) and (counterMax.min() + counterMin.min() < 3):
            system.integrate(horizon=1e4, timeStep=1e-1)

        # Follow trajectory
        system.currentState = nextStep(state_t1, system.systemDynamics, timeStep, method="RK4")

        # If a steady state is reached: return the steady state
        reachedSteadyState = system.steadyStatesReached([state_t0, state_t1, system.currentState], tolerence, printMessage=True)
        if reachedSteadyState is not None:
            return np.array([reachedSteadyState, reachedSteadyState, reachedSteadyState, np.zeros(system.dim)])

        # Look for local extrema for each population
        isLocalMax = (state_t0 <= state_t1) * (system.currentState <= state_t1)
        isLocalMin = (state_t1 <= state_t0) * (state_t1 <= system.currentState)

        for pop in range(system.dim):
            # If local max
            if isLocalMax[pop]:
                if abs((state_t1[pop] - localMax[pop]) <= tolerence * max(localMax[pop], EPSILON)):
                    counterMax[pop] += 1
                else:
                    counterMax[pop] = 1
                    localMax[pop] = state_t1[pop]

            # If local min
            if isLocalMin[pop]:
                if abs((state_t1[pop] - localMin[pop]) <= tolerence * max(localMin[pop], EPSILON)):
                    counterMin[pop] += 1
                else:
                    counterMin[pop] = 1
                    localMin[pop] = state_t1[pop]

        # Check is convergence criteria is reached
        limitCycleReached = (counterMax.min() >= convergenceCriteria) & (counterMin.min() >= convergenceCriteria)
        isNonZeroExtremalPop = (isLocalMax | isLocalMin) * (state_t1 > EPSILON)

        if limitCycleReached & isNonZeroExtremalPop.any():
            # Filter on population that cannot be used as reference in period assessment
            # that is, constant population. The steady state should only be 0 (no other constant value is filtered)
            isLocalMinAndCandidate = isLocalMin * isNonZeroExtremalPop
            isLocalMaxAndCandidate = isLocalMax * isNonZeroExtremalPop
            t0 = time.time()

            oscillationAnalysis = system.getAverageLevelAndPeriod(isLocalMinAndCandidate, isLocalMaxAndCandidate, state_t1, timeStep, maxIteration, tolerence)

            if oscillationAnalysis is None:
                return np.array([localMax, localMin, np.zeros(system.dim), (-1)*np.ones(system.dim)])

            [averagePopLevel, oscillationPeriod] = oscillationAnalysis

            t1 = time.time()
            print("Oscillatory trajectory. Average population level and period found in %fs" %(t1-t0))
            return np.array([localMax, localMin, averagePopLevel, oscillationPeriod * np.ones(system.dim)])

        state_t0 = state_t1
        state_t1 = system.currentState
        nbIteration += 1

    print("no extrema could be found with the given tolerence")
    return np.array([system.currentState, state_t0, state_t1, (-1)*np.ones(system.dim)])

def readPermanentStatesFromExcel(fileName, popName, a_sample):
    results = None
    for pop, name in popName.items():
       resDataFrame = pandas.read_excel(fileName, sheetname=name)
       resDataFrame.index = np.round(resDataFrame.index,2)
       resDataFrame = resDataFrame.loc[a_sample][['max', 'min', 'mean', 'period']]
       if results is None:
           results = np.empty((resDataFrame.index.size, resDataFrame.columns.size, len(popName.keys())))
       results[:,:,pop] = resDataFrame.values
    return results, resDataFrame.index.values

def bifurcationCartography(coupledSystem, a_sample, b_sample, resultsDir, timeStep, horizon, checkExisting=None):
    numberTimeStep = int(horizon / timeStep)
    # Loop over parameter b
    for b in b_sample:
        coupledSystem.update({"b":b}, currentState=coupledSystem.initialState)

        # Initialize outputs
        statistics = np.empty((a_sample.size,4,4))
        initialStates = np.empty((a_sample.size,4))
        times = np.zeros((a_sample.size,2))

        # Get results excel workbook name
        name = resultsDir + "results_"
        for param, value in sorted(coupledSystem.parameters.items(), key=(lambda x: {"a":0, "b":1, "e":2, "d": 3, "m":4}[x[0]])):
            if param == "a": continue
            name += "_" + param + str(int(value * 100)).zfill(2)
        name += ".xlsx"

        # Ship if results file already exists and option is turned on
        if checkExisting == SKIP_IF_EXISTING:
            if os.path.isfile(name):
                continue
        # Re-use previous results but complete than if necessary
        elif checkExisting is not None:
            if checkExisting == USE_PREVIOUS_RESULTS:
                oldResultsFile = name
            else:
                oldResultsFile = name.replace(".xlsx",checkExisting+".xlsx")
            if os.path.isfile(oldResultsFile):
                results_old, a_sample_old = readPermanentStatesFromExcel(oldResultsFile, coupledSystem.popName)

        # Loop over parameter a
        for i,a in enumerate(a_sample):
            t1 = time.time()
            print('\nComputing for b= %f and a = %f ' %(b,a))
            coupledSystem.update({"a":a})

            # Get previous results if option is turned on and results exists for current a and b values
            if os.path.isfile(oldResultsFile) and (checkExisting is not None):
                if round(a,2) in np.round(a_sample_old, 2):
                    index = list(np.round(a_sample_old, 2)).index(round(a,2))
                    if not np.isnan(results_old[index]).any():
                        statistics[i] = results_old[index]
#                        times[i] = times_old[index]
                        continue

            # Pre-treatment to accelerate convergence
            t2 = time.time()
            coupledSystem.initialState = coupledSystem.currentState + np.maximum(np.random.random_sample(4) * coupledSystem.currentState * 0.01, 1e-3)
            initialStates[i] = coupledSystem.initialState
            coupledSystem.integrate(horizon=5e3, timeStep=1e-1)
            t3 = time.time()

            # Compute permanent dynamics min/mean/max values for current a/b parameters
            res = getPermanentExtrema(coupledSystem, timeStep, convergenceCriteria=10, tolerence=1e-5, maxIteration=numberTimeStep)

            t4 = time.time()

            # Stored results
            statistics[i] = res
            times[i] = [t4-t1, t3-t2]
            print('finished in %f' %times[i,0])

        # Write results to excel books using pandas
        writer = pandas.ExcelWriter(name)
        durationStat = pandas.DataFrame(index=a_sample, columns=["Overall duration", "Preprossessing duration"])
        durationStat["Overall duration"] = pandas.Series(times[:,0], index=a_sample)
        durationStat["Preprossessing duration"] = pandas.Series(times[:,1], index=a_sample)
        for pop in range(coupledSystem.dim):
            results = pandas.DataFrame(index=a_sample, columns=['max','min','mean','period'])
            for j,h in enumerate(results.columns):
                results[h] = pandas.Series(statistics[:, j, pop], index=a_sample)

            results.to_excel(writer, sheet_name = coupledSystem.popName[pop])

        durationStat.to_excel(writer, sheet_name = "calculation time")

        writer.save()
