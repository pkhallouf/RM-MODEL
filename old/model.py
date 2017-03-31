# -*- coding: utf-8 -*-
import numericalAnalysis
import numpy as np
import scipy
import math

class Rosenzweig_MacArthur:
    def __init__(self, initialState, parameters):
        self.dim = len(initialState)
        self.initialState = initialState
        self.parameters = parameters

    def preyDynamics(self, state=None):
        """
        a = 1 / carryingCapacity
        e = 1 / dynCoefficient
        x = preyState = state[0]
        y = predatorState = state[1]
        returns ( 1 / e ) * [ x * (1 - a * x) - x * y / (1 + x) ]
        """
        if state is None :
            state = self.initialState
        preyState = state[0]
        predatorState = state[1]
        carryingCapacity = 1 / self.parameters["a"]
        dynCoefficient = 1 / self.parameters["e"]
        return dynCoefficient * (preyState * (1 - (1/carryingCapacity) * preyState) - preyState * predatorState / (1 + preyState))

    def predatorDynamics(self, state=None):
        """
        x = preyState
        y = predatorState
        m = predatorMortality
        return x * y / (1 + x) - m * y
        """
        if state is None:
            state = self.initialState
        preyState = state[0]
        predatorState = state[1]
        predatorMortality = self.parameters["m"]
        return preyState * predatorState / (1 + preyState) - predatorMortality * predatorState

    def systemDynamics(self, state=None):
        if state is None:
            state = self.initialState

        return np.array([self.preyDynamics(state), self.predatorDynamics(state)])

    def getTrajectory(self, horizon, timeStep, initialState=None, method='RK4'):
        if initialState is None:
            initialState = self.initialState
        numberTimeStep = int(horizon/timeStep)
        return numericalAnalysis.explicitSolver(self.systemDynamics, initialState, numberTimeStep, timeStep, method='RK4')

class Coupled_RM:
    def __init__(self,initialState, parameters):
        self.dim = len(initialState)
        self.initialState = initialState
        self.parameters = parameters
        self.subSystems = [
        Rosenzweig_MacArthur(initialState[0:2], {"a":parameters["a"], "e":parameters["e"], "m":parameters["m"], "d":parameters["d"]}),
        Rosenzweig_MacArthur(initialState[2:4], {"a":parameters["b"], "e":parameters["e"], "m":parameters["m"], "d":parameters["d"]})
        ]

    def migration(self, state=None, migrationOperator=None):
        if state is None:
            tuple(sys.initialState for sys in self.subSystems)
            state = np.concatenate(tuple(sys.initialState for sys in self.subSystems), axis=0)
        if migrationOperator is not None:
            raise ValueError("Using another migration operator is not implemented")
        else:
            delta = (state[0]-state[2])
            return np.array([-self.subSystems[0].parameters["d"] * delta, self.subSystems[1].parameters["d"] * delta])

    def systemDynamics(self, state=None):
        if state is None:
            state = self.initialState
        coupledSubSystemsDynamics = np.append(self.subSystems[0].systemDynamics(state[0:2]), self.subSystems[1].systemDynamics(state[2:4]))
        coupledSubSystemsDynamics[0:4:2] += self.migration(state)
        return coupledSubSystemsDynamics

    def getTrajectory(self, horizon, timeStep, initialState=None, method='RK4'):
        if initialState is None:
            initialState = self.initialState
        numberTimeStep = int(horizon/timeStep)
        return numericalAnalysis.explicitSolver(self.systemDynamics, initialState, numberTimeStep, timeStep, method='RK4')

    def f(self, x, param):
        e = self.parameters["e"]
        d = self.parameters["d"]
        return ((1-e*d)/(2*param)) * (1+math.sqrt(1+((4*param*e*d*x)/((1-e*d)**2))))

    def getSteadyStates_0(self):
        return np.zeros(4)

    def getSteadyStates_1(self):
        a = self.parameters["a"]
        b = self.parameters["b"]
        g = lambda state: state - np.array([self.f(state[2], a), 0, self.f(state[0], b), 0])
        return scipy.optimize.fsolve(g,[1,1,1,1])

    def getSteadyStates_2(self):
        a = self.parameters["a"]
        b = self.parameters["b"]
        e = self.parameters["e"]
        d = self.parameters["d"]
        m = self.parameters["m"]

        x = self.f(m/(1-m), a)
        u = m/(1-m)
        y = 0
        v = (1/m)*(u*(1-b*u)+e*d*(x-u))
        return np.array([x, y, u, v])

    def getSteadyStates_3(self):
        a = self.parameters["a"]
        b = self.parameters["b"]
        e = self.parameters["e"]
        d = self.parameters["d"]
        m = self.parameters["m"]

        x = m/(1-m)
        u = self.f(m/(1-m), b)
        y = (1/m)*(x*(1-a*x)+e*d*(u-x))
        v = 0
        return np.array([x, y, u, v])

    def getSteadyStates_4(self):
        a = self.parameters["a"]
        b = self.parameters["b"]
        m = self.parameters["m"]

        x = m/(1-m)
        y = (1-(1+a)*m)/((1-m)**2)
        u = m/(1-m)
        v = (1-(1+b)*m)/((1-m)**2)
        return np.array([x, y, u, v])

    def getSteadyStates(self):
        steadyStates = [self.getSteadyStates_0(),
        self.getSteadyStates_1(),
        self.getSteadyStates_2(),
        self.getSteadyStates_3(),
        self.getSteadyStates_4()]
        return np.array(steadyStates)

def preyDynamics(preyState, predatorState, carryingCapacity, dynCoefficient):
    """
    a = 1 / carryingCapacity
    e = 1 / dynCoefficient
    x = preyState
    y = predatorState
    return ( 1 / e ) * [ x * (1 - a * x) - x * y / (1 + x) ]
    """
    return dynCoefficient * (preyState * (1 - (1/carryingCapacity) * preyState) - preyState * predatorState / (1 + preyState))

def predatorDynamics(preyState, predatorState, predatorMortality):
    """
    x = preyState
    y = predatorState
    m = predatorMortality
    return x * y / (1 + x) - m * y
    """
    return preyState * predatorState / (1 + preyState) - predatorMortality * predatorState

def systemDynamics(state, carryingCapacity, predatorMortality, dynCoefficient):
    return np.array([preyDynamics(state[0], state[1], carryingCapacity, dynCoefficient), predatorDynamics(state[0], state[1], predatorMortality)])

def systemDynamics2(state, t, carryingCapacity, predatorMortality, dynCoefficient):
    return np.array([preyDynamics(state[0], state[1], carryingCapacity, dynCoefficient), predatorDynamics(state[0], state[1], predatorMortality)])

def systemDynamics3(t, state, carryingCapacity, predatorMortality, dynCoefficient):
    return np.array([preyDynamics(state[0], state[1], carryingCapacity, dynCoefficient), predatorDynamics(state[0], state[1], predatorMortality)])

def migration(initialState, systemDynamics1, systemDynamics2, rate):
    delta = initialState[0] - initialState[2]
    systemDynamics1[0] -= rate * delta
    systemDynamics2[0] += rate * delta
    return np.append(systemDynamics1,systemDynamics2)

def coupledSystemDynamics(state, parameters):
    systemDynamics1 = systemDynamics(state[0:2], 1 / parameters["a"], parameters["m"], 1 / parameters["e"])
    systemDynamics2 = systemDynamics(state[2:4], 1 / parameters["b"], parameters["m"], 1 / parameters["e"])
    return migration(state, systemDynamics1, systemDynamics2, parameters["d"])

def coupledSystemDynamics2(state, t, a, b, m, d, e):
    systemDynamics1 = systemDynamics(state[0:2], 1/a, m, e)
    systemDynamics2 = systemDynamics(state[2:4], 1/b, m, e)
    return migration(systemDynamics1, systemDynamics2, d)

def coupledSystemDynamics3(t, state, a, b, m, d, e):
    systemDynamics1 = systemDynamics(state[0:2], 1/a, m, e)
    systemDynamics2 = systemDynamics(state[2:4], 1/b, m, e)
    return migration(systemDynamics1, systemDynamics2, d)

def jacobian2(state, t, carryingCapacity, mortality):
    e = 1
    a = 1/carryingCapacity
    b = 0
    m = mortality
    d = 0
    jac = [[0]*2]*2
    jac[0][0] = (-2*a*state[0]-state[1]/(1+state[0])**2+1)/e - d
    jac[0][1] = -(1/e)*state[0]/(1+state[0])
#    jac[0][2] = d
    jac[1][0] = state[1]/(1+state[0])**2
    jac[1][1] = state[0]/(1+state[0]) - m
#    jac[2][0] = d
#    jac[2][2] = (-2*b*state[2]-state[3]/(1+state[2])**2+1)/e - d
#    jac[2][3] = -(1/e)*state[2]/(1+state[2])
#    jac[3][2] = state[3]/(1+state[2])**2
#    jac[3][3] = state[2]/(1+state[2]) - m
    return jac

def jacobian3(t, state, a, b, m, d, e):
    jac = np.zeros((4,4))
    jac[0,0] = (-2*a*state[0]-state[1]/(1+state[0])**2+1)/e - d
    jac[0,1] = -(1/e)*state[0]/(1+state[0])
    jac[0,2] = d
    jac[1,0] = state[1]/(1+state[0])**2
    jac[1,1] = state[0]/(1+state[0]) - m
    jac[2,0] = d
    jac[2,2] = (-2*b*state[2]-state[3]/(1+state[2])**2+1)/e - d
    jac[2,3] = -(1/e)*state[2]/(1+state[2])
    jac[3,2] = state[3]/(1+state[2])**2
    jac[3,3] = state[2]/(1+state[2]) - m
    return jac
