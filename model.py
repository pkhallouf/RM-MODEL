# -*- coding: utf-8 -*-
import numericalAnalysis
import numpy as np
import scipy
import scipy.optimize
import math
import copy

class Rosenzweig_MacArthur:
    def __init__(self, initialState, parameters):
        self.dim = len(initialState)
        self.initialState = copy.copy(initialState)
        self.parameters = copy.copy(parameters)

    def update(self, parameters=None, initialState=None):
        if initialState is not None:
            self.initialState = copy.copy(initialState)
        if parameters is not None:
            self.parameters.update(parameters)

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
        self.initialState = copy.copy(initialState)
        self.currentState = copy.copy(initialState)
        self.parameters = copy.copy(parameters)
        self.paramOrder = {"a":1, "b":2, "m":3, "d":4, "e":5}
        self.subSystems = [
        Rosenzweig_MacArthur(self.initialState[0:2], {"a":self.parameters["a"], "e":self.parameters["e"], "m":self.parameters["m"], "d":self.parameters["d"]}),
        Rosenzweig_MacArthur(self.initialState[2:4], {"a":self.parameters["b"], "e":self.parameters["e"], "m":self.parameters["m"], "d":self.parameters["d"]})
        ]
        self.popIndex = {'x':0, 'y':1, 'u':2, 'v': 3}
        self.popName  = {self.popIndex[key]:key for key in self.popIndex.keys()}

    def update(self, parameters=None, initialState=None, currentState=None):
        if parameters != None:
            self.parameters.update(parameters)
            self.subSystems[0].update({key:parameters[key] for key in parameters if key in self.subSystems[0].parameters})
            self.subSystems[1].update({key:parameters[key] for key in parameters if key in self.subSystems[1].parameters and key != "a"})
            if "b" in parameters:
                self.subSystems[1].update({"a":parameters["b"]})
        if initialState is not None:
            self.initialState = copy.copy(initialState)
        if currentState is not None:
            self.currentState = copy.copy(currentState)

    def migration(self, state=None, migrationOperator=None):
        delta = (state[0]-state[2])
        return np.array([-self.subSystems[0].parameters["d"] * delta, 0, self.subSystems[1].parameters["d"] * delta, 0])

    def systemDynamics(self, state=None):
        if state is None:
            state = self.initialState
        coupledSubSystemsDynamics = np.append(self.subSystems[0].systemDynamics(state[0:2]), self.subSystems[1].systemDynamics(state[2:4]))
        coupledSubSystemsDynamics += self.migration(state)
        return coupledSubSystemsDynamics

    def getTrajectory(self, horizon, timeStep, initialState=None, method='RK4'):
        if initialState is None:
            initialState = self.initialState
        numberTimeStep = int(horizon/timeStep)
        return numericalAnalysis.explicitSolver(self.systemDynamics, initialState, numberTimeStep, timeStep, method='RK4')

    def integrate(self, horizon, timeStep, initialState=None, method='RK4'):
        if initialState is not None:
            self.currentState = initialState
        numberTimeStep = int(horizon/timeStep)
        for t in xrange(numberTimeStep):
            self.currentState = numericalAnalysis.nextStep(self.currentState, self.systemDynamics, timeStep=timeStep, method=method)

    def flow(self, horizon, timeStep, initialState=None, method='RK4'):
        if initialState is None:
            initialState = self.initialState
        state = initialState
        numberTimeStep = int(horizon/timeStep)
        for t in xrange(numberTimeStep):
            state = numericalAnalysis.nextStep(state, self.systemDynamics, timeStep=timeStep, method=method)
        return state

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

    def getSteadyStatesOutOfParam(self, parametersDict, updateParam=False):
        """
        Returns steady states relative to parametersDict values
        if updateParam is False, reset object parameter attribute to previous one
        """
        oldParam = copy(self.parameters)
        self.update(parametersDict)
        steadyStates = self.getSteadyStates()
        if updateParam == False:
            self.update(oldParam)
        return steadyStates

    def steadyStatesReached(self, successiveStatesList, tolerence, printMessage=False):
        for i, steadyState in enumerate(self.getSteadyStates()):
            reached = True
            for state in successiveStatesList:
                reached = reached & (abs(steadyState - state) <= tolerence * np.maximum(steadyState, numericalAnalysis.EPSILON)).all()
            if reached:
                if printMessage: print("Trajectory converges toward steady state %i" %i)
                return steadyState
        return None

    def getAverageLevelAndPeriod(self, isLocalMax, isLocalMin, iniState, timeStep, maxIteration, tolerence):
        """
        isLocalMax and isLocalMin must apply to iniState
        """
        # Define is which population in the reference state to use as a reference and whether it is minimal or maximal
        checkMax, checkMin = False, False
        maximalPop = np.where(isLocalMax)[0]

        if maximalPop.size == 0:
            minimalPop = np.where(isLocalMin)[0]
#            candidate = [popIndex for popIndex in minimalPop if iniState[popIndex] > numericalAnalysis.EPSILON]
            if minimalPop.size == 0:
                print("None of the populations in iniState was extremal")
                return None
            else:
                pop = minimalPop[0]
                checkMin = True
        else:
#            candidate = [popIndex for popIndex in maximalPop if iniState[popIndex] > numericalAnalysis.EPSILON]
            pop = maximalPop[0]
            checkMax = True

        # Initialize trajectory and indicators
        state_t1 = copy.copy(iniState)
        nbTimeStepBetweenExtrema = np.zeros(5)
        nbTimeStepBetweenExtrema[0] = 0
        cumul = iniState + self.currentState
        condition = False
        counter = 0
        referenceState = None

        # Go through trajectory
        while counter < 5:
            state_t0 = state_t1
            state_t1 = self.currentState
            self.currentState = numericalAnalysis.nextStep(self.currentState, self.systemDynamics, timeStep, method="RK4")
            if counter >= 1:
                cumul += self.currentState
                nbTimeStepBetweenExtrema[counter] += 1
            condition = checkMax * (state_t0[pop] <= state_t1[pop]) * (self.currentState[pop] <= state_t1[pop])
            condition = condition | (checkMin * (state_t0[pop] >= state_t1[pop]) * (self.currentState[pop] >= state_t1[pop]))
#            if referenceState is not None:
#                condition = condition & ((abs(referenceState - self.currentState) / (np.maximum(numericalAnalysis.EPSILON, referenceState))).max() < tolerence)
            if condition:
                counter += 1
#                if referenceState is None : referenceState = self.currentState
            if nbTimeStepBetweenExtrema.sum() >= maxIteration:
                print("Average population level period and could not be found...")
                return None

        averagePopLevel = cumul / (nbTimeStepBetweenExtrema.sum())
        oscillationPeriod = (nbTimeStepBetweenExtrema.sum()) * timeStep / counter
#        print("nb time step between extrema: ", nbTimeStepBetweenExtrema)
        return np.array([averagePopLevel, oscillationPeriod * np.ones(self.dim)])


