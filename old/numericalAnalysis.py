# -*- coding: utf-8 -*-
import time
from scipy.integrate import odeint, ode
import numpy as np

def nextStep(initialState, dynamics, timeStep, method="RK4", detect_0_crossingEvent=False):
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
    else:
        raise ValueError("Numerical resolution method %s could not be found" % method)
    if detect_0_crossingEvent & (newState.min() < 0):
        newState_corrected = initialState
        newTimeStep = timeStep / 10
        nbNewIter = 10
        print(initialState, newState, timeStep, newTimeStep, nbNewIter)
        for i in range(nbNewIter):
            newState_corrected = nextStep(newState_corrected, dynamics, newTimeStep, method)
        return newState_corrected
    return newState


def explicitSolver(dynamics, initialState, numberTimeStep, timeStep, method='RK4'):
    """
    F(X1) = F(X0) + dF(X0) * (X1 - X0)
    """
    dim = len(initialState)
    trajectory = np.zeros((numberTimeStep,dim))
    trajectory[0] = initialState
    for t in range(1, numberTimeStep):
        trajectory[t] = nextStep(trajectory[t-1], dynamics, timeStep, method)
    return trajectory


def flow(dynamics, initialState, horizon, timeStep, numberTimeStep, method='RK4'):
    """
    horizon [int]: number of time steps for which solution will calculated
    numberTimeStep [int]: number of time steps that will be kept in the returned trajectory
    """
    dim = len(initialState)
    trajectory = np.zeros((numberTimeStep,dim))
    state = initialState
    t = 0
    if not horizon >= numberTimeStep:
        e = ValueError('Cannot compute flow with horizon < numberTimeStep')
        raise e
    while 1:
        state = nextStep(state, dynamics, timeStep, method, detect_0_crossingEvent = False)
        if t >= horizon-numberTimeStep: break
        else : t+= 1
    trajectory[0] = state
    for t in xrange(1,numberTimeStep):
        trajectory[t] = nextStep(trajectory[t-1], dynamics, timeStep, method, detect_0_crossingEvent = False)
    return trajectory


def scipyODEsolvers(initialState, parameters, timeStep, numberTimeStep):
    """
    Deprecated
    scipy.ode does not work correctly: system.t seems fixed...
    """
    timeSample = timeStep * np.array(range(numberTimeStep))
    t0 = time.time()
    results = odeint(systemDynamics2, initialState[0:2], timeSample, (1 / parameters["a"], parameters["m"]), printmessg=True, full_output=1)
    t1 = time.time()
    if results[1]['message'] == 'Integration successful.':
        print "scipy.integration.odeint successfully finished in %f" %(t1-t0)
        return results[0]
    ### If scipy.integration.odeint unsuccessfull
    t=0
    trajectory_ode = np.empty((numberTimeStep,len(initialState)))
    trajectory_ode[0] = initialState[0:2]
    system = ode(systemDynamics3)
    system.set_initial_value(initialState[0:2], 0)
    system.set_f_params(1 / parameters["a"], parameters["m"])
    t2 = time.time()
    while 1:
        t += 1
        state_t = system.integrate(system.t+timeStep)
        if system.successful():
            trajectory_ode[t] = state_t
        else:
            print "integration failed at time step %f, number %f -> using first order Euler method" %(system.t+timeStep, t)
            state_t = nextStep(trajectory_ode[t-1], lambda state : systemDynamics(state, 1 / parameters["a"], parameters["m"]), timeStep)
            trajectory_ode[t] = state_t
            system.t += timeStep
        if t >= numberTimeStep-1:
            t3 = time.time()
            print "integration loop successfully finished in %f" %(t3-t2)
            return trajectory_ode


def getPermanentExtrema(system, timeStep=1e-5, convergenceCriteria=15, tolerence=1e-5, maxIteration=1e6):
    dim = len(system.initialState)
    counterMax = np.array([0]*dim)
    counterMin = np.array([0]*dim)
    localMax = np.array([-10.0]*dim)
    localMin = np.array([-10.0]*dim)

    steadyStates = system.getSteadyStates()

    state_t0 = system.initialState
    state_t1 = nextStep(state_t0, system.systemDynamics, timeStep, method="RK4")

    nbIteration = 0
    while nbIteration < maxIteration:
        state_t2 = nextStep(state_t1, system.systemDynamics, timeStep, method="RK4")

        # If a steady state is reached: return the steady state
        for steadyState in steadyStates:
            if (abs(steadyState - state_t2).max() < tolerence):
                print("Trajectory converges toward steady state")
                return np.array([steadyState, steadyState])

        # Look for local extrema for each population
        isLocalMax = (state_t0 <= state_t1) * (state_t2 <= state_t1)
        isLocalMin = (state_t1 <= state_t0) * (state_t1 <= state_t2)
        for pop in range(dim):

            # If local max
            if isLocalMax[pop]:
                if abs((state_t1[pop] - localMax[pop])/localMax[pop]) < tolerence:
                    counterMax[pop] += 1
                else:
                    counterMax[pop] = 1
                    localMax[pop] = state_t1[pop]

            # If local min
            elif isLocalMin[pop]:
                if abs((state_t1[pop] - localMin[pop])/localMin[pop]) < tolerence:
                    counterMin[pop] += 1
                else:
                    counterMin[pop] = 1
                    localMin[pop] = state_t1[pop]

        # If criteria reached
        if (counterMax.min() >= convergenceCriteria) & (counterMin.min() >= convergenceCriteria):
            print("Oscillatory trajectory")
            return np.array([localMax, localMin])

        state_t0 = state_t1
        state_t1 = state_t2
        nbIteration += 1

    print("no extrema could be found with the given tolerence")
    return np.array([counterMax, counterMax])
