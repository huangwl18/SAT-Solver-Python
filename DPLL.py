# Implementation of the Davis-Putnam-Logemann-Loveland (DPLL) algorithm in Python with NumPy dependency
# If the formula is satisfiable, it will return 'sat' and print out the assignment mapping
# If the formula is not satisfiable, it will return 'unsat' and print out the last learned clause
# The assignment mapping is a 1d NumPy array:
# the indices represent the variables, where 0 represents variable 1, and so on
# the values represent the mappings which can be either 0 or 1

import numpy as np

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
    formula = np.array([clause + [0] * (length-len(clause)) for clause in formula], dtype=int)
    # delete all rows that only contain zeros
    formula = formula[np.any(formula != 0, axis=1)]
    # find the number of variables in f
    num_var = np.amax(abs(formula))
    return (formula, num_var)

def PureLiteralElimination(_f, _num_var, assignment):
    # create a copy of input
    f = np.copy(_f)
    num_var = _num_var
    # loop through every variable
    for i in range(1, num_var + 1):
        # check if the variable exists in only one polarity
        num_positive = np.argwhere(f == i).size
        num_negative = np.argwhere(f == -i).size
        # if the variable only exists in negative polarity
        if(num_positive == 0 or num_negative == 0):
            # eliminate all clauses containing this variable
            f = np.delete(f, np.argwhere(np.abs(f) == i)[:, 0], axis=0)
            # update assignment map
            if(num_positive == 0):
                assignment[i - 1] = 0
            if(num_negative == 0):
                assignment[i - 1] = 1
    return f

def findUnitClauses(_f):
    # create a copy of input
    f = np.copy(_f)
    return f[np.count_nonzero(f, axis=1) == 1]

def UnitPropagation(_f, _unitCls, assignment):
    #print '_f', _f
    #print '_unitCls', _unitCls
    f = np.copy(_f)
    unitCls = np.copy(_unitCls)
    # iterate through all unit clauses given
    for i in range(unitCls.shape[0]):
        # extract the variable in the current unit clause
        value = unitCls[i, np.argwhere(unitCls[i] != 0)]
        # update assignment map
        assignment[np.abs(value) - 1] = np.greater(value, 0).astype(np.int)
        # get the index of rows in formula that contain the current variable
        rowContainedValue = np.any(np.equal(f, value), axis=1)
        # only use the formula that does not contain these rows
        f = f[np.invert(rowContainedValue)]
        # get the index of elements in formula that contain the opposite sign of the current variable
        elementsContainedNegativeValue = np.argwhere(f == -value)
        # set these elements to zeros
        f[elementsContainedNegativeValue[:,0], elementsContainedNegativeValue[:,1]] = 0
    # if cannot find more unit clause to propagate, return current formula
    if(findUnitClauses(f).size == 0):
        return f
    return UnitPropagation(f, findUnitClauses(f), assignment)

def MakeDecision(_f):
    f_flatten = np.ravel(_f)
    f_flatten = f_flatten[np.not_equal(f_flatten, 0)]
    temp = np.array([np.zeros(_f.shape[1])])
    temp[0,0] = np.bincount(np.abs(f_flatten)).argmax()
    return temp.astype(np.int)

def runDPLLwithDIMACSfile(file_path):
    formula, num_var = readDIMACS(file_path)
    assignment = np.zeros([num_var]) - 1
    def DPLL(_f, _num_var):
        # check conflict (check if there is any row that is empty)
        if (_f[np.count_nonzero(_f, axis=1) == 0].size > 0):
            return "unsat"
        # check satisfied
        if (_f.size == 0):
            # print assignment
            return "sat"
        unitClauses = findUnitClauses(_f)
        # if find unit clauses, do unit propagation
        if (unitClauses.size > 0):
            _f_copy = UnitPropagation(_f, unitClauses, assignment)
            return DPLL(_f_copy, _num_var)
        puredFormula = PureLiteralElimination(_f, num_var, assignment)
        if (not np.array_equal(_f, puredFormula)):
            return DPLL(puredFormula, _num_var)
        decision = MakeDecision(_f)
        if (DPLL(np.concatenate((_f, decision), axis=0), _num_var) == "sat"):
            return "sat"
        else:
            return DPLL(np.concatenate((_f, -decision), axis=0), _num_var)
    print DPLL(formula, num_var)
    print 'assignment\n', assignment