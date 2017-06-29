# linsolve

[![Build Status](https://travis-ci.org/HERA-Team/linsolve.svg?branch=master)](https://travis-ci.org/HERA-Team/linsolve)
[![Coverage Status](https://coveralls.io/repos/github/HERA-Team/linsolve/badge.svg?branch=master)](https://coveralls.io/github/HERA-Team/linsolve?branch=master)

linsolve is a module providing high-level tools for linearizing and solving systems of equations.

# Overview

The solvers in linsolve (LinearSolver, LogProductSolver, and LinProductSolver), 
generally follow the form:
```python
data = {'a1*x+b1*y': np.array([5.,7]), 'a2*x+b2*y': np.array([4.,6])}
ls = LinearSolver(data, a1=1., b1=np.array([2.,3]), a2=2., b2=np.array([1.,2]))
sol = ls.solve()
```

where equations are passed in as a dictionary where each key is a string
describing the equation (which is parsed according to python syntax) and each
value is the corresponding "measured" value of that equation.  Variable names
in equations are checked against keyword arguments to the solver to determine
if they are provided constants or parameters to be solved for.  Parameter anmes
and solutions are return are returned as key:value pairs in ls.solve().
Parallel instances of equations can be evaluated by providing measured values
as numpy arrays.  Constants can also be arrays that comply with standard numpy
broadcasting rules.  Finally, weighting is implemented through an optional wgts
dictionary that parallels the construction of data.

LinearSolver solves linear equations of the form `'a*x + b*y + c*z'`.
LogProductSolver uses logrithms to linearize equations of the form `'x*y*z'`.
LinProductSolver uses symbolic Taylor expansion to linearize equations of the
form `'x*y + y*z'`.

# Package Details
## Known Issues and Planned Improvements

For details see the [issue log](https://github.com/HERA-Team/linsolve/issues).

## Community Guidelines
Contributions to this package to add new file formats or address any of the
issues in the [issue log](https://github.com/HERA-Team/linsolve/issues) are very welcome.
Please submit improvements as pull requests against the repo after verifying that
the existing tests pass and any new code is well covered by unit tests.

Bug reports or feature requests are also very welcome, please add them to the
issue log after verifying that the issue does not already exist.
Comments on existing issues are also welcome.

# Installation
## Dependencies
First install dependencies. 

* numpy >= 1.10
* scipy

## Install linsolve
For simple installation, the latest stable version is available via pip using ```pip install linsolve```

### Optionally install the development version
For the development version, clone the repository using
```git clone https://github.com/HERA-Team/linsolve.git```

Navigate into the directory and run ```python setup.py install```.
Note that this will automatically install any missing dependencies. If you use anaconda or another package manager you might prefer to first install the dependencies as described above.

## Tests
From the source linsolve directory run ```python tests/linsolve_test.py```.

