from model import *
from numericalAnalysis import *
import matplotlib.pyplot as plt
from scipy.integrate import odeint, ode
import time
import sys
import numpy as np


initialState = np.array([1, 1, 1, 1])
parameters = {
"a": 0.2,
"b": 0.3,
"e": 0.5,
"m": 0.4,
"d": 0.5
}

print "running simulation with parameters", parameters, " and initialState ", initialState

timeStep = 1e-4
horizon = 1e3
numberTimeStep = int(horizon / timeStep)
#timeSample = timeStep * np.array(range(numberTimeStep))

#isolatedSystem = Rosenzweig_MacArthur(initialState[0:2], parameters)
coupledSystem = Coupled_RM(initialState, parameters)

#isolatedTraj = isolatedSystem.getTrajectory(horizon, timeStep)
#coupledTraj = coupledSystem.getTrajectory(horizon, timeStep)

#sysDyn = lambda state : coupledSystemDynamics(state, parameters)
#t0 = time.time()
#trajectory_e = explicitSolver(sysDyn, initialState, numberTimeStep, timeStep, method='RK4')
#t1 = time.time()

#sys.exit('ended')

a_first, a_last, da = 0.2, 1.50, 0.01
extrema = np.empty(((a_last-a_first)/da,4,2))
times = np.zeros((a_last-a_first)/da)
for i,a_times_100 in enumerate(xrange(int(a_first/da), int(a_last/da))):
    t1 = time.time()
    a = a_times_100 * 0.01
    print('Computing for a = %f ' %a)
    parameters["a"] = a
    res = getPermanentExtrema(coupledSystem, timeStep, convergenceCriteria=3, tolerence=2e-2, maxIteration=numberTimeStep)
    extrema[i] = np.transpose(res)
    t2= time.time()
    times[i] = t2-t1
    print('finished in %f' %(t2-t1))
#
#sol2 = solution(sol[len(sol)-1], parameters, timeStep, numberTimeStep)
t3 = time.time()

sys.exit('ended')
t0 = time.time()
#trajectory = eulerSchemeResolution(lambda state : systemDynamics(state, 1 / parameters["a"], parameters["m"]), initialState[0:2], numberTimeStep, timeStep)
#trajectory = flow(lambda state : systemDynamics(state, 1 / parameters["a"], parameters["m"]), initialState[0:2], horizon = horizon, timeStep = timeStep, numberTimeStep = numberTimeStep)

t1 = time.time()
#print "trajectory endend in %s sec" %(t1 - t0)

t2 = time.time()
#print("tail endend in %s sec" %(t2 - t1))

#trajectory_odeint = odeint(systemDynamics2, initialState[0:2], timeSample, (1 / parameters["a"], parameters["m"]), printmessg=True)
t3 = time.time()
print("odeint endend in %s sec" %(t3 - t2))

steadyState = getSteadyState(parameters)

t4 = time.time()
print("ode endend in %s sec" %(t4 - t3))