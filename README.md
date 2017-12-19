# SAT-Solver-Python

This repository contains the implementation of the SAT (Boolean Satisfiability) solvers, DPLL and CDCL.

Both are implemented in Python with NumPy dependency (probably not the most efficient way, but fun!)

The solvers currently only accepts CNF files in DIMACS format. Just pass in the file directory (and maximum runtime for CDCL) to runDPLLwithDIMACSfile (or runCDCLwithDIMACSfile) and have fun!

See the comments in each class for the explanations of the output.
