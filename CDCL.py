# Implementation of the Conflict-Driven Clause Learning (CDCL) algorithm in Python with NumPy dependency
# Currently only accept cnf file in DIMACS format
# Call runCDCLwithDIMACSfile by providing the directory path to the cnf file and a specified maximum runtime
# If the formula is satisfiable, it will return 'sat' and print out the assignment mapping
# If the formula is not satisfiable, it will return 'unsat' and print out the last learned clause
# If the runtime exceeds the maximum runtime, it will return 'unknown' and print out the current assignment mapping
# The assignment mapping is a 2d NumPy array,
# first row: variable names
# second row: assignment mappings (0 or 1)
# third row: reason clauses (-1 representing decided variable, > 0 representing the index of the reason clause)
# forth row: decision levels

import numpy as np
import time

def readDIMACS(cnf_file):
    f = open(cnf_file, "r")
    formula = list()
    for line in f:
        l = line.split()
        if len(l) != 0 and l[0] != 'c' and l[0] != 'p':
            l.pop()
            formula.append(l)
    f.close()
    # convert list of lists to 2d numpy array
    length = len(sorted(formula, key=len, reverse=True)[0])
    formula = np.array([clause + [0] * (length - len(clause)) for clause in formula], dtype=int)
    # delete all rows that only contain zeros
    formula = formula[np.any(formula != 0, axis=1)]
    # find the number of variables in f
    num_var = np.amax(abs(formula))
    return (formula, num_var)


def PureLiteralElimination(_f, _num_var, _a):
    # create a copy of input
    f = np.copy(_f)
    num_var = _num_var
    a = np.copy(_a)
    # loop through every variable
    for i in range(1, num_var + 1):
        # check if the variable exists in only one polarity
        num_positive = np.argwhere(f == i).size
        num_negative = np.argwhere(f == -i).size
        # if the variable only exists in negative polarity
        if (num_positive == 0 or num_negative == 0):
            # eliminate all clauses containing this variable
            f = np.delete(f, np.argwhere(np.abs(f) == i)[:, 0], axis=0)
            # update assignment map
            if (num_positive == 0):
                a = np.concatenate((a, np.array([[i], [0], [-1], [0]])), axis=1)
            if (num_negative == 0):
                a = np.concatenate((a, np.array([[i], [1], [-1], [0]])), axis=1)
    return f, a


# f, a: formula and assignment passed in
# return only the unit clause in the formula and its clause index
def PickClause(_f, _a):
    # create copies for passed references
    f = np.copy(_f)
    a = np.copy(_a)
    if (a.size == 0):
        # find the index of unit clauses in f
        unitClsIndex = np.count_nonzero(f, axis=1) == 1
        f = f[unitClsIndex]
        if f[f != 0].size == 0:
            return None, None
        else:
            return np.squeeze(f[f != 0]), np.squeeze(np.argwhere(unitClsIndex == True)[0])

            # return np.squeeze(f[unitClsIndex][f[unitClsIndex] != 0][0]), np.squeeze(np.argwhere(unitClsIndex == True)[0])
    else:
        # add one column to formula to indicate the index of clause
        f = np.concatenate((np.expand_dims(np.arange(f.shape[0]), axis=1), f), axis=1)
        # check each clause if it contains any positive literals which have already been assigned True
        positiveTrue = np.any(np.isin(f[:, 1:], a[0, a[1, :] == 1]), axis=1)
        # check each clause if it contains any negative literals which have already been assigned False
        negativeFalse = np.any(np.isin(f[:, 1:], -a[0, a[1, :] == 0]), axis=1)
        # merge the above two to get indices of the clauses that have not been satisfied
        notSatisfiedIndex = np.invert(np.logical_or(positiveTrue, negativeFalse))
        # update formula
        f = f[notSatisfiedIndex]
        # check each literal if it is of positive polarity but has already been assigned False
        positiveFalse = np.isin(f[:, 1:], a[0, a[1, :] == 0])
        # update formula by making these literals 0
        f[:, 1:][positiveFalse] = 0
        # check each literal if it is of negative polarity but has already been assigned True
        negativeTrue = np.isin(f[:, 1:], -a[0, a[1, :] == 1])
        # update formula by making these literals 0
        f[:, 1:][negativeTrue] = 0
        conflictClauses = f[np.count_nonzero(f[:, 1:], axis=1) == 0]
        if conflictClauses.size > 0:
            conflictClauses = conflictClauses[0, 0]
        if (conflictClauses.size > 0):
            return np.squeeze(np.array([0])), np.squeeze(conflictClauses)
        unitClsIndex = np.count_nonzero(f[:, 1:], axis=1) == 1
        # update formula so that it only contains unit clauses
        f = f[unitClsIndex]
        if f[:, 1:][f[:, 1:] != 0].size == 0:
            return None, None
        else:
            return np.squeeze(f[:, 1:][f[:, 1:] != 0][0]), np.squeeze(f[:, 0][0])


def UnitPropagationCDCL(_f, _a, _d):
    f = np.copy(_f)
    a = np.copy(_a)
    while True:
        cls, clsIndex = PickClause(f, a)
        # if no unit clause found
        if cls == None:
            return a
        # if conflict
        elif cls == 0:
            a = np.concatenate((a, np.array([[-1], [-1], [clsIndex], [_d]])), axis=1)
            return a
        # else: update assignment map
        else:
            if cls > 0:
                a = np.concatenate((a, np.array([[cls], [1], [clsIndex], [_d]])), axis=1)
            elif cls < 0:
                a = np.concatenate((a, np.array([[-cls], [0], [clsIndex], [_d]])), axis=1)
            else:
                print 'error in UnitPropagationCDCL'


def AnalyzeAndLearn(_f, _a):
    conflictIndex = np.argwhere(_a[0, :] == -1)[0]
    conflictCls = _f[_a[2, conflictIndex], :]
    reasonQueue = np.empty([2, 0])
    for i in range(conflictCls[conflictCls != 0].size):
        reasonQueue = np.concatenate((reasonQueue, np.array([[conflictCls[0, i]], [-1]])), axis=1).astype(np.int)
        indexInAssignment = np.squeeze(np.argwhere(_a[0, :] == np.abs(conflictCls[0, i])))
        if indexInAssignment.size > 1:
            print 'error in AnalyzeAndLearn'
        decisionLevel = _a[3, indexInAssignment]
        reasonQueue[1, -1] = decisionLevel
    while (np.argwhere(reasonQueue[1, :] == np.amax(reasonQueue[1, :])).size > 1):
        firstMaxIndex = np.ravel(np.argwhere(reasonQueue[1, :] == np.amax(reasonQueue[1, :])))[-1]
        count = 1
        while (True):
            count = count + 1
            literal = reasonQueue[0, firstMaxIndex]
            indexInAssignment = np.argwhere(_a[0, :] == np.abs(literal))
            if indexInAssignment.size > 1:
                print 'error in AnalyzeAndLearn 1'
            reasonClsIndex = np.squeeze(_a[2, indexInAssignment])
            if (reasonClsIndex == -1):
                firstMaxIndex = np.ravel(np.argwhere(reasonQueue[1, :] == np.amax(reasonQueue[1, :])))[-count]
                continue
            reasonCls = _f[reasonClsIndex]
            break
        reasonQueue = np.delete(reasonQueue, firstMaxIndex, axis=1)
        for i in range(reasonCls.size):
            if np.abs(reasonCls[i]) != np.abs(literal) and reasonCls[i] != 0 and np.isin(np.abs(reasonCls[i]),
                                                                                         np.abs(reasonQueue[0, :]),
                                                                                         invert=True):
                reasonQueue = np.concatenate((reasonQueue, np.array([[reasonCls[i]], [-1]])), axis=1)
                indexInAssignment = np.squeeze(np.argwhere(_a[0, :] == np.abs(reasonCls[i])))
                if indexInAssignment.size > 1:
                    print 'error in AnalyzeAndLearn 2'
                decisionLevel = _a[3, indexInAssignment]
                reasonQueue[1, -1] = decisionLevel
        # condition 1: if all literals are decided at level = 0, return unsat
        if np.all(reasonQueue[1, :] == 0):
            return np.array([-1]), np.array([-1])
        # condition 2: if only one is decided at level > 0, return assembled clause and jump back to 0
        elif reasonQueue[1, :].size == 1:
            return reasonQueue[0, :], 0
    # condition 3: if all literals are decided at levels > 0, return assembled clause and second highest decision level
    if np.all(reasonQueue[1, :] >= 0):
        return reasonQueue[0, :], np.partition(reasonQueue[1, :], -2)[-2]
    else:
        print 'error in AnalyzeAndLearn 3'
        return np.array([-2]), np.array([-2])


def MakeDecision(_f, _a, _d):
    # flatten the formula for choosing random element
    f_flatten = np.ravel(_f)
    a = np.copy(_a)
    d = np.copy(_d)
    # remove all elements that are zero
    f_flatten = f_flatten[np.not_equal(f_flatten, 0)]
    variableChosen = np.abs(np.random.choice(f_flatten))
    while (np.isin(variableChosen, _a[0, :])):
        variableChosen = np.squeeze(np.abs(np.random.choice(f_flatten)))
    a = np.concatenate((a, np.array([[variableChosen], [1], [-1], [d + 1]])), axis=1)
    return a, d + 1


def CDCL(formula, num_var, max_runtime_in_seconds):
    assignment = np.empty([4, 0])
    formula, assignment = PureLiteralElimination(formula, num_var, assignment)
    decisionLevel = 0
    startTime = time.time()
    while time.time() - startTime < max_runtime_in_seconds:
        iteration = iteration + 1
        assignment = UnitPropagationCDCL(formula, assignment, decisionLevel).astype(np.int)
        # if there is no conflict
        if np.argwhere(assignment[0, :] == -1).size == 0:
            # if all variables are assigned
            if np.all(np.isin(np.arange(num_var) + 1, assignment[0, :])):
                print 'assignment: \n', assignment
                return 'sat'
            # if there is still variable not assigned (and no unit clause found), make a decision
            else:
                assignment, decisionLevel = MakeDecision(formula, assignment, decisionLevel)
                assignment = assignment.astype(np.int)
        else:
            learnedClause, newDecisionLevel = AnalyzeAndLearn(formula, assignment)
            if newDecisionLevel < 0:
                print 'last learned clause: \n', learnedClause
                return 'unsat'
            else:
                if learnedClause.size <= formula.shape[1]:
                    temp = np.zeros(formula.shape[1])
                    for i in range(learnedClause.size):
                        temp[i] = learnedClause[i]
                    formula = np.concatenate((formula, np.expand_dims(temp, axis=0).astype(np.int)), axis=0)
                else:
                    difference = learnedClause.size - formula.shape[1]
                    formula = np.concatenate((formula, np.zeros([formula.shape[0], difference])), axis=1)
                    formula = np.concatenate((formula, np.expand_dims(learnedClause, axis=0)), axis=0)
                assignment = assignment[:, assignment[3, :] <= newDecisionLevel]
                decisionLevel = newDecisionLevel
    print 'current decision literal stack: \n', assignment
    return 'unknown'

def runCDCLwithDIMACSfile(file_path, max_runtime_in_seconds):
    formula, num_var = readDIMACS(file_path)
    print CDCL(formula, num_var, max_runtime_in_seconds)