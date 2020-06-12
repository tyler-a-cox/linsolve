import linsolve
import numpy as np
import time, random

np.random.seed(0)

NPRMS = 10
NEQS = 100
SIZE = 100000
#MODE = 'solve' # 1.56 s
MODE = 'lsqr' # 5.4 s
#MODE = 'pinv' # 3.4 s

prms = {'g%d' % i: np.arange(SIZE) for i in range(NPRMS)}
prm_list = list(prms.keys())
prms['c0'] = np.arange(SIZE)

eqs = [('+c0*'.join(['%s'] * 5))  % tuple(random.sample(prm_list, 5)) 
            for i in range(NEQS)]

data = {eq: eval(eq, prms) for eq in eqs}

ls = linsolve.LinearSolver(data, c0=prms['c0'])
t0 = time.time()
sol = ls.solve(mode=MODE)
t1 = time.time()

print('Solved in {}'.format(t1-t0))

for k in prm_list:
    assert np.mean(np.abs(sol[k] - prms[k])**2) < 1e-3
