'''Module providing high-level tools for linearizing and finding chi^2 minimizing 
solutions to systems of equations.

Solvers: LinearSolver, LogProductSolver, and LinProductSolver.

These generally follow the form:
> data = {'a1*x+b1*y': np.array([5.,7]), 'a2*x+b2*y': np.array([4.,6])}
> ls = LinearSolver(data, a1=1., b1=np.array([2.,3]), a2=2., b2=np.array([1.,2]))
> sol = ls.solve()

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

LinearSolver solves linear equations of the form 'a*x + b*y + c*z'.
LogProductSolver uses logrithms to linearize equations of the form 'x*y*z'.
LinProductSolver uses symbolic Taylor expansion to linearize equations of the
form 'x*y + y*z'.

For more detail on usage, see linsolve_example.ipynb
'''

from __future__ import absolute_import, division, print_function
import numpy as np
import ast
from scipy.sparse import lil_matrix, csr_matrix
import scipy.sparse.linalg
import warnings
from copy import deepcopy
from functools import reduce

def ast_getterms(n):
    '''Convert an AST parse tree into a list of terms.  E.g. 'a*x1+b*x2' -> [[a,x1],[b,x2]]'''
    if type(n) is ast.Name: return [[n.id]]
    elif type(n) is ast.Num: return [[n.n]]
    elif type(n) is ast.Expression: return ast_getterms(n.body)
    elif type(n) is ast.UnaryOp:
        assert(type(n.op) is ast.USub)
        return [[-1]+ast_getterms(n.operand)[0]]
    elif type(n) is ast.BinOp:
        if type(n.op) is ast.Mult:
            return [ast_getterms(n.left)[0] + ast_getterms(n.right)[0]]
        elif type(n.op) is ast.Add: return ast_getterms(n.left) + ast_getterms(n.right)
        elif type(n.op) is ast.Sub: return ast_getterms(n.left) + [[-1] + ast_getterms(n.right)[0]]
        else: raise ValueError('Unsupported operation: %s' % str(n.op))
    else: raise ValueError('Unsupported: %s' % str(n))

def get_name(s, isconj=False):
    '''Parse variable names of form 'var_' as 'var' + conjugation.'''
    if not type(s) is str:
        if isconj: return str(s), False
        else: return str(s)
    if isconj: return s.rstrip('_'), s.endswith('_') # tag names ending in '_' for conj
    else: return s.rstrip('_') # parse 'name_' as 'name' + conj


class Constant:
    '''Container for constants (which can be arrays) in linear equations.'''
    def __init__(self, name, constants):
        self.name = get_name(name)
        if type(name) is str: self.val = constants[self.name]
        else: self.val = name
        try: self.dtype = self.val.dtype
        except(AttributeError): self.dtype = type(self.val)
    def shape(self):
        try: return self.val.shape
        except(AttributeError): return ()
    def get_val(self, name=None):
        '''Return value of constant. Handles conj if name='varname_' is requested 
        instead of name='varname'.'''
        if name is not None and type(name) is str:
            name, conj = get_name(name, isconj=True)
            assert(self.name == name)
            if conj: return self.val.conjugate()
            else: return self.val
        else: return self.val


class Parameter:
    
    def __init__(self, name):
        '''Container for parameters that are to be solved for.'''
        self.name = get_name(name)

    def sparse_form(self, name, eqnum, prm_order, prefactor, re_im_split=True):
        xs,ys,vals = [], [], []
        # separated into real and imaginary parts iff one of the variables is conjugated with "_"
        if re_im_split: 
            name,conj = get_name(name, True)
            ordr,ordi = 2*prm_order[self.name], 2*prm_order[self.name]+1 
            cr,ci = prefactor.real, prefactor.imag
            i = 2*eqnum
            # (cr,ci) * (pr,pi) = (cr*pr-ci*pi, ci*pr+cr*pi)
            xs.append(i); ys.append(ordr); vals.append(cr) # real component
            xs.append(i+1); ys.append(ordr); vals.append(ci) # imag component
            if not conj:
                xs.append(i); ys.append(ordi); vals.append(-ci) # imag component
                xs.append(i+1); ys.append(ordi); vals.append(cr) # imag component
            else:
                xs.append(i); ys.append(ordi); vals.append(ci) # imag component
                xs.append(i+1); ys.append(ordi); vals.append(-cr) # imag component
        else:
            xs.append(eqnum); ys.append(prm_order[self.name]); vals.append(prefactor)
        return xs, ys, vals
    
    def get_sol(self, x, prm_order):
        '''Extract prm value from appropriate row of x solution.'''
        if x.shape[0] > len(prm_order): # detect that we are splitting up real and imaginary parts
            ordr,ordi = 2*prm_order[self.name], 2*prm_order[self.name]+1
            return {self.name: x[ordr] + np.complex64(1.0j)*x[ordi]}
        else: return {self.name: x[prm_order[self.name]]}


class LinearEquation:
    '''Container for all prms and constants associated with a linear equation.'''
    def __init__(self, val, **kwargs):
        self.val = val
        if type(val) is str:
            n = ast.parse(val, mode='eval')
            val = ast_getterms(n)
        self.wgts = kwargs.pop('wgts',np.float32(1.))
        self.has_conj = False
        constants = kwargs.pop('constants', kwargs)
        self.process_terms(val, constants)

    def process_terms(self, terms, constants):
        '''Classify terms from parsed str as Constant or Parameter.'''
        self.consts, self.prms = {}, {}
        for term in terms:
            for t in term:
                try:
                    self.add_const(t, constants)
                except(KeyError): # must be a parameter then
                    p = Parameter(t)
                    self.has_conj |= get_name(t,isconj=True)[-1] # keep track if any prms are conj
                    self.prms[p.name] = p
        self.terms = self.order_terms(terms)

    def add_const(self, name, constants):
        '''Manually add a constant of given name to internal list of constants. Value is drawn from constants.'''
        n = get_name(name)
        if n in constants and isinstance(constants[n], Constant): c = constants[n]
        else: c = Constant(name, constants) # raises KeyError if not a constant
        self.consts[c.name] = c
    
    def order_terms(self, terms):
        '''Reorder terms to obey (const1,const2,...,prm) ordering.'''
        for L in terms: L.sort(key=lambda x: get_name(x) in self.prms)
        # Validate that each term has exactly 1 unsolved parameter.
        for t in terms:
            assert(get_name(t[-1]) in self.prms)
            for ti in t[:-1]:
                assert(type(ti) is not str or get_name(ti) in self.consts)
        return terms

    def eval_consts(self, const_list, wgts=np.float32(1.)):
        '''Multiply out constants (and wgts) for placing in matrix.'''
        const_list = [self.consts[get_name(c)].get_val(c) for c in const_list]
        return wgts**.5 * reduce(lambda x,y: x*y, const_list, np.float32(1.))
        # this has the effect of putting the square root of the weights into each A matrix
        #return 1. * reduce(lambda x,y: x*y, const_list, 1.)

    def sparse_form(self, eqnum, prm_order, re_im_split=True):
        '''Returns the row and col information and the values of coefficients to build up 
        part of the sparse (CSR) reprentation of the A matrix corresponding to this equation.'''
        xs, ys, vals = [], [], []
        for term in self.terms:
            p = self.prms[get_name(term[-1])]
            f = self.eval_consts(term[:-1], self.wgts)
            x,y,val = p.sparse_form(term[-1], eqnum, prm_order, f.flatten(), re_im_split)
            xs += x; ys += y; vals += val
        return xs, ys, vals
    
    def eval(self, sol):
        '''Given dict of parameter solutions, evaluate this equation.'''
        rv = 0
        for term in self.terms:
            total = self.eval_consts(term[:-1])
            name,isconj = get_name(term[-1],isconj=True)
            if isconj: total *= np.conj(sol[name])
            else: total *= sol[name]
            rv += total
        return rv
        

def verify_weights(wgts, keys):
    '''Given wgts and keys, ensure wgts have all keys and are all real.
    If wgts == {} or None, return all 1s.'''
    if wgts is None or wgts == {}:
        return {k: np.float32(1.) for k in keys}
    else:
        for k in keys:
            assert(k in wgts) # must have weights for all keys
            assert(np.iscomplexobj(wgts[k]) == False) # tricky errors happen if wgts are complex
        return wgts

def infer_dtype(values):
    '''Given a list of values, return the appropriate data 
    type for matrices, solutions.  
    Returns float32, float64, complex64, or complex128'''
    # determine the data type of all numpy arrays
    types = [v.dtype for v in values if hasattr(v,'dtype')]
    types = [t for t in types if type(t) is np.dtype]
    # ensure we are at least a float32 if we were passed integers
    types.append(np.float32(1).dtype)
    # if any python constants are complex, promote to complex, but otherwise
    # don't promote to double if we have floats/doubles/ints in python
    py_types = [v.dtype if hasattr(v,'dtype') else type(v) for v in values]
    if type(1j) in py_types: types.append(np.complex64(1).dtype)
    dtype = reduce(np.promote_types, types)
    return dtype


class LinearSolver:

    def __init__(self, data, wgts={}, sparse=False, **kwargs):
        """Set up a linear system of equations of the form 1*a + 2*b + 3*c = 4.

        Args:
            data: Dictionary that maps linear equations, written as valid python-interpetable strings 
                that include the variables in question, to (complex) numbers or numpy arrarys. 
                Variables with trailing underscores '_' are interpreted as complex conjugates.
            wgts: Dictionary that maps equation strings from data to real weights to apply to each 
                equation. Weights are treated as 1/sigma^2. All equations in the data must have a weight 
                if wgts is not the default, {}, which means all 1.0s.
            sparse: Boolean (default False). If True, handles all matrix algebra with sparse matrices. 
                May be faster for certain systems of equations. 
            **kwargs: keyword arguments of constants (python variables in keys of data that 
                are not to be solved for)

        Returns:
            None
        """
        self.data = data
        self.keys = list(data.keys())
        self.sparse = sparse
        self.wgts = verify_weights(wgts, self.keys)
        constants = kwargs.pop('constants', kwargs)
        self.eqs = [LinearEquation(k,wgts=self.wgts[k], constants=constants) for k in self.keys]
        # XXX add ability to have more than one measurment for a key=equation
        self.prms = {}
        for eq in self.eqs: 
            self.prms.update(eq.prms)
        self.consts = {}
        for eq in self.eqs: 
            self.consts.update(eq.consts) 
        self.prm_order = {}
        for i,p in enumerate(self.prms): 
            self.prm_order[p] = i

        # infer dtype for later arrays
        self.re_im_split = kwargs.pop('re_im_split',False)
        #go through and figure out if any variables are conjugated
        for eq in self.eqs: 
            self.re_im_split |= eq.has_conj
        self.dtype = infer_dtype(list(self.data.values()) + list(self.consts.values()) + list(self.wgts.values()))
        if self.re_im_split: self.dtype = np.real(np.ones(1, dtype=self.dtype)).dtype
        self.shape = self._shape()

    def _shape(self):
        '''Get broadcast shape of constants, weights for last dim of A'''
        sh = []
        for k in self.consts:
            shk = self.consts[k].shape()
            if len(shk) > len(sh): sh += [0] * (len(shk)-len(sh))
            for i in range(min(len(sh),len(shk))): sh[i] = max(sh[i],shk[i])
        for k in self.wgts:
            try: shk = self.wgts[k].shape
            except(AttributeError): continue
            if len(shk) > len(sh): sh += [0] * (len(shk)-len(sh))
            for i in range(min(len(sh),len(shk))): sh[i] = max(sh[i],shk[i])
        return tuple(sh)

    def _A_shape(self):
        '''Get shape of A matrix (# eqs, # prms, data.size). Now always 3D.'''
        try: 
            sh = (reduce(lambda x,y: x*y, self.shape),) # flatten data dimensions so A is always 3D
        except(TypeError): 
            sh = (1,)
        if self.re_im_split: 
            return (2*len(self.eqs),2*len(self.prm_order))+sh
        else: return (len(self.eqs),len(self.prm_order))+sh

    def get_A(self):
        '''Return A matrix for A*x=y.'''
        A = np.zeros(self._A_shape(), dtype=self.dtype)
        xs,ys,vals = self.sparse_form()
        ones = np.ones_like(A[0,0])
        #A[xs,ys] += [v * ones for v in vals] # This is broken when a single equation has the same param more than once
        for x,y,v in zip(xs,ys,[v * ones for v in vals]):
            A[x,y] += v # XXX ugly
        return A

    def sparse_form(self):
        '''Returns a lists of lists of row and col numbers and coefficients in order to
        express the linear system as a CSR sparse matrix.'''
        xs, ys, vals = [], [], []
        for i,eq in enumerate(self.eqs):
            x,y,val = eq.sparse_form(i, self.prm_order, self.re_im_split)
            # print 'val:', val
            xs += x; ys += y; vals += val
        return xs, ys, vals

    def get_A_sparse(self):
        '''Fixes dimension needed for CSR sparse matrix representation.'''
        xs,ys,vals = self.sparse_form()
        ones = np.ones(self._A_shape()[2:],dtype=self.dtype)
        for n,val in enumerate(vals): 
            if not isinstance(val, np.ndarray) or val.size == 1:
                vals[n] = ones*val
        return np.array(xs), np.array(ys), np.array(vals).T
    
    def get_weighted_data(self):
        '''Return y = data * wgt**.5 as a 2D vector, regardless of original data/wgt shape.'''
        d = np.array([self.data[k] for k in self.keys])
        if len(self.wgts) > 0:
            w = np.array([self.wgts[k] for k in self.keys])
            w.shape += (1,) * (d.ndim-w.ndim)
            d.shape += (1,) * (w.ndim-d.ndim)
            d = d*(w**.5) 
            # this is w**.5 because A already has a factor of w**.5 in it, so 
            # (At N^-1 A)^1 At N^1 y ==> (At A)^1 At d (where d is the result of this 
            # function and A is redefined to include half of the weights)
        self._data_shape = d.shape[1:] # store for reshaping sols to original
        d.shape = (d.shape[0],-1) # Flatten 
        if self.re_im_split:
            rv = np.empty((2*d.shape[0],)+d.shape[1:], dtype=self.dtype)
            rv[::2],rv[1::2] = d.real, d.imag
            return rv
        else: return d
    
    def _invert_default_sparse(self, A, y, rcond):
        '''Use the lsqr inverter, but if an LinAlgError is encountered, try using pinv.'''
        try:
            xhat = self._invert_lsqr_sparse(A, y, rcond)
        except(np.linalg.LinAlgError):
            xhat = self._invert_pinv_sparse(A, y, rcond)
        return xhat

    def _invert_lsqr_sparse(self, A, y, rcond):
        '''Use the scipy.sparse lsqr solver.'''
        xhat = scipy.sparse.linalg.lsqr(A, y[...,0], atol=rcond, btol=rcond)[0]
        xhat.shape += (1,)
        return xhat

    def _invert_pinv_sparse(self, A, y, rcond):
        '''Use pinv to invert AtA matrix.  Tends to be ~10x slower than lsqr for sparse matrices'''
        At = A.T.conj()
        AtA = At.dot(A).toarray()
        try: AtAi = np.linalg.pinv(AtA, rcond=rcond)
        except(np.linalg.LinAlgError): AtAi = np.linalg.inv(AtA)
        return AtAi.dot(At.dot(y)) 

    def _invert_solve_sparse(self, A, y, rcond):
        '''Use linalg.solve to solve a fully constrained (non-degenerate) system of equations.
        Tends to be ~3x slower than lsqr for sparse matrices.  'rcond' is unused, but passed
        as an argument to match the interface of other _invert methods.'''
        At = A.T.conj()
        AtA = At.dot(A).toarray()
        Aty = At.dot(y) # automatically dense bc y is dense
        #xhat = scipy.sparse.linalg.spsolve(AtA, Aty)
        return np.linalg.solve(AtA, Aty)

    def _invert_lsqr(self, A, y, rcond):
        '''Use np.linalg.lstsq to solve a system of equations.  Usually the best 
        performer, but for a fully-constrained system, 'solve' can be faster.  Also,
        there are a couple corner cases where lstsq is unstable but pinv works 
        for the same rcond. It seems particularly the case for single precision matrices.'''
        xhat = np.linalg.lstsq(A, y[...,0], rcond=rcond)[0]
        xhat.shape += (1,)
        return xhat

    def _gen_AtAiAt(self, A, rcond):
        '''Helper function for forming (At A)^-1 At.  Uses pinv to invert.'''
        At = A.T.conj()
        AtA = At.dot(A)
        # finding inverse is about 3x slower than solve
        try: AtAi = np.linalg.pinv(AtA, rcond=rcond)
        except(np.linalg.LinAlgError): 
            AtAi = np.linalg.inv(AtA)
        AtAiAt = AtAi.dot(A.T.conj()) 
        return AtAiAt

    def _invert_pinv(self, A, y, rcond):
        '''Use np.linalg.pinv to invert AtA matrix.  Tends to be about ~3x slower than solve.'''
        AtAiAt = self._gen_AtAiAt(A, rcond)
        return np.dot(AtAiAt,y)

    def _invert_solve(self, A, y, rcond):
        '''Use np.linalg.solve to solve a system of equations.  Requires a fully constrained
        system of equations (i.e. doesn't deal with singular matrices).  Can by ~1.5x faster that lstsq
        for this case. 'rcond' is unused, but passed as an argument to match the interface of other
        _invert methods.'''
        At = A.T.conj()
        AtA = At.dot(A)
        Aty = At.dot(y)
        return np.linalg.solve(AtA, Aty) # is supposed to error if singular, but doesn't seem to.

    def _invert_default(self, A, y, rcond):
        '''Use the lsqr inverter, but if an LinAlgError is encountered, try using pinv.'''
        try:
            xhat = self._invert_lsqr(A, y, rcond)
        except(np.linalg.LinAlgError):
            xhat = self._invert_pinv(A, y, rcond)
        return xhat

    def solve(self, rcond=1e-10, mode='default', verbose=False):
        """Compute x' = (At A)^-1 At * y, returning x' as dict of prms:values.

        Args:
            rcond: cutoff ratio for singular values useed in numpy.linalg.lstsq, numpy.linalg.pinv,
                or (if sparse) as atol and btol in scipy.sparse.linalg.lsqr
            mode: 'default', 'lsqr', 'pinv', or 'solve', selects which inverter to use. 
                'default': tries 'lsqr' but if a LinAlgError is encountered, backs off to try 'pinv'.
                'lsqr': uses numpy.linalg.lstsq to do an inversion-less solve.  Usually 
                    the fastest solver.
                'solve': uses numpy.linalg.solve to do an inversion-less solve.  Fastest, 
                    but only works for fully constrained systems of equations.
                'pinv': uses numpy.linalg.pinv to perform a pseudo-inverse and then solves.  Can
                    sometimes be more numerically stable (but slower) than 'lsqr'.
                All of these modes are superceded if the same system of equations applies
                to all datapoints in an array.  In this case, a inverse-based method is used so
                that the inverted matrix can be re-used to solve all array indices.
            verbose: print information about iterations

        Returns:
            sol: a dictionary of solutions with variables as keys
        """
        assert(mode in ['default','lsqr','pinv','solve'])
        y = self.get_weighted_data()
        Ashape = self._A_shape()
        x = np.empty((Ashape[1],y.shape[-1]), dtype=self.dtype)
        if self.sparse:
            if mode == 'default': _invert = self._invert_default_sparse
            elif mode == 'lsqr': _invert = self._invert_lsqr_sparse
            elif mode == 'pinv': _invert = self._invert_pinv_sparse
            elif mode == 'solve': _invert = self._invert_solve_sparse
            xs, ys, vals = self.get_A_sparse()
            for k in range(y.shape[-1]):
                if verbose: print('Solving %d/%d' % (k, y.shape[-1]))
                Ak = csr_matrix((vals[min(k,Ashape[-1]-1)], (xs,ys))) 
                x[...,k:k+1] = _invert(Ak, y[...,k:k+1], rcond)
        else: 
            A = self.get_A()
            assert(A.ndim == 3)
            if Ashape[-1] == 1 and y.shape[-1] > 1: # can reuse inverse
                A = A[...,0]
                AtAiAt = self._gen_AtAiAt(A, rcond)
                x = np.dot(AtAiAt, y)
                # XXX old way below factor of ~4 slower. All tests
                # pass without. Necessary?
                #for k in range(y.shape[-1]):
                #    if verbose: print('Solving %d/%d' % (k, y.shape[-1]))
                #    x[...,k:k+1] = np.dot(AtAiAt,y[...,k:k+1])
            else: # we can't reuse inverses
                if mode == 'default': _invert = self._invert_default
                elif mode == 'lsqr': _invert = self._invert_lsqr
                elif mode == 'pinv': _invert = self._invert_pinv
                elif mode == 'solve': _invert = self._invert_solve
                for k in range(y.shape[-1]):
                    if verbose: print('Solving %d/%d' % (k, y.shape[-1]))
                    Ak = A[...,min(k,Ashape[-1]-1)]
                    x[...,k:k+1] = _invert(Ak, y[...,k:k+1], rcond)

        x.shape = x.shape[:1] + self._data_shape # restore to shape of original data
        sol = {}
        for p in list(self.prms.values()): sol.update(p.get_sol(x,self.prm_order))
        return sol

    def eval(self, sol, keys=None):
        """Returns a dictionary evaluating data keys to the current values given sol and consts.
        Uses the stored data object unless otherwise specified."""
        if keys is None: keys = self.keys
        elif type(keys) is str: keys = [keys]
        elif type(keys) is dict: keys = list(keys.keys())
        result = {}
        for k in keys:
            eq = LinearEquation(k, **self.consts)
            result[k] = eq.eval(sol)
        return result
    
    def _chisq(self, sol, data, wgts, evaluator):
        """Internal adaptable chisq calculator."""
        if len(wgts) == 0: sigma2 = {k: 1.0 for k in list(data.keys())} #equal weights
        else: sigma2 = {k: wgts[k]**-1 for k in list(wgts.keys())} 
        evaluated = evaluator(sol, keys=data)
        chisq = 0
        for k in list(data.keys()): chisq += np.abs(evaluated[k]-data[k])**2 / sigma2[k]
        return chisq
    
    def chisq(self, sol, data=None, wgts=None):
        """Compute Chi^2 = |obs - mod|^2 / sigma^2 for the specified solution. Weights are treated as 1/sigma^2. 
        wgts = {} means sigma = 1. Default uses the stored data and weights unless otherwise overwritten."""
        if data is None: 
            data = self.data
        if wgts is None: 
            wgts = self.wgts
        wgts = verify_weights(wgts, list(data.keys()))
        return self._chisq(sol, data, wgts, self.eval)
        

# XXX need to add support for conjugated constants...maybe this already works because we have conjugated constants inherited from taylor expansion
def conjterm(term, mode='amp'):
    '''Modify prefactor for conjugated terms, according to mode='amp|phs|real|imag'.'''
    f = {'amp':1,'phs':-1,'real':1,'imag':1j}[mode] # if KeyError, mode was invalid
    terms = [[f,t[:-1]] if t.endswith('_') else [t] for t in term]
    return reduce(lambda x,y: x+y, terms)

def jointerms(terms): 
    '''String that joins lists of lists of terms as the sum of products.'''
    return '+'.join(['*'.join(map(str,t)) for t in terms])


class LogProductSolver: 

    def __init__(self, data, wgts={}, sparse=False, **kwargs):
        """Set up a nonlinear system of equations of the form a*b = 1.0 to linearze via logarithm.

        Args:
            data: Dictionary that maps nonlinear product equations, written as valid python-interpetable 
                strings that include the variables in question, to (complex) numbers or numpy arrarys. 
                Variables with trailing underscores '_' are interpreted as complex conjugates (e.g. x*y_ 
                parses as x * y.conj()).
            wgts: Dictionary that maps equation strings from data to real weights to apply to each 
                equation. Weights are treated as 1/sigma^2. All equations in the data must have a weight 
                if wgts is not the default, {}, which means all 1.0s.
            sparse: Boolean (default False). If True, handles all matrix algebra with sparse matrices. 
                May be faster for certain systems of equations. 
            **kwargs: keyword arguments of constants (python variables in keys of data that 
                are not to be solved for)

        Returns:
            None
        """
        keys = list(data.keys())
        wgts = verify_weights(wgts, keys)
        eqs = [ast_getterms(ast.parse(k, mode='eval')) for k in keys]
        logamp, logphs = {}, {}
        logampw, logphsw = {}, {}
        for k,eq in zip(keys,eqs):
            assert(len(eq) == 1) # equations have to be purely products---no adds
            eqamp = jointerms([conjterm([t],mode='amp') for t in eq[0]])
            eqphs = jointerms([conjterm([t],mode='phs') for t in eq[0]])
            dk = np.log(data[k])
            logamp[eqamp],logphs[eqphs] = dk.real, dk.imag
            try: logampw[eqamp],logphsw[eqphs] = wgts[k], wgts[k]
            except(KeyError): pass
        constants = kwargs.pop('constants', kwargs)
        self.dtype = infer_dtype(list(data.values()) + list(constants.values()) + list(wgts.values()))
        logamp_consts, logphs_consts = {}, {}
        for k in constants:
            c = np.log(constants[k]) # log unwraps complex circle at -pi
            logamp_consts[k], logphs_consts[k] = c.real, c.imag
        self.ls_amp = LinearSolver(logamp, logampw, sparse=sparse, constants=logamp_consts)
        self.ls_phs = LinearSolver(logphs, logphsw, sparse=sparse, constants=logphs_consts)

    def solve(self, rcond=1e-10, mode='default', verbose=False):
        """Solve both amplitude and phase by taking the log of both sides to linearize.

        Args:
            rcond: cutoff ratio for singular values useed in numpy.linalg.lstsq, numpy.linalg.pinv,
                or (if sparse) as atol and btol in scipy.sparse.linalg.lsqr
            mode: 'default', 'lsqr', 'pinv', or 'solve', selects which inverter to use. 
                'default': tries 'lsqr' but if a LinAlgError is encountered, backs off to try 'pinv'.
                'lsqr': uses numpy.linalg.lstsq to do an inversion-less solve.  Usually 
                    the fastest solver.
                'solve': uses numpy.linalg.solve to do an inversion-less solve.  Fastest, 
                    but only works for fully constrained systems of equations.
                'pinv': uses numpy.linalg.pinv to perform a pseudo-inverse and then solves.  Can
                    sometimes be more numerically stable (but slower) than 'lsqr'.
                All of these modes are superceded if the same system of equations applies
                to all datapoints in an array.  In this case, a inverse-based method is used so
                that the inverted matrix can be re-used to solve all array indices.
            verbose: print information about iterations

        Returns:
            sol: a dictionary of complex solutions with variables as keys
        """
        sol_amp = self.ls_amp.solve(rcond=rcond, mode=mode, verbose=verbose)
        sol_phs = self.ls_phs.solve(rcond=rcond, mode=mode, verbose=verbose)
        sol = {}
        for k in sol_amp:
            with warnings.catch_warnings():
                # ignore warnings about casting complex to float, which only happens
                # if there were no imaginary data to deal with.
                warnings.simplefilter('ignore', np.ComplexWarning)
                sol[k] = np.exp(sol_amp[k] + 
                    np.complex64(1j)*sol_phs[k]).astype(self.dtype)
        return sol

def taylor_expand(terms, consts={}, prepend='d'):
    '''First-order Taylor expand terms (product of variables or the sum of a 
    product of variables) wrt all parameters except those listed in consts.'''
    taylors = []
    for term in terms: taylors.append(term)
    for term in terms:
        for i,t in enumerate(term):
            if type(t) is not str or get_name(t) in consts: continue
            taylors.append(term[:i]+[prepend+t]+term[i+1:])
    return taylors


# XXX make a version of linproductsolver that taylor expands in e^{a+bi} form
class LinProductSolver:

    def __init__(self, data, sol0, wgts={}, sparse=False, **kwargs):
        """Set up a nonlinear system of equations of the form a*b + c*d = 1.0 
        to linearize via Taylor expansion and solve iteratively using the Gauss-Newton algorithm.

        Args:
            data: Dictionary that maps nonlinear product equations, written as valid python-interpetable 
                strings that include the variables in question, to (complex) numbers or numpy arrarys. 
                Variables with trailing underscores '_' are interpreted as complex conjugates (e.g. x*y_ 
                parses as x * y.conj()).
            sol0: Dictionary mapping all variables (as keyword strings) to their starting guess values.
                This is the point that is Taylor expanded around, so it must be relatively close to the
                true chi^2 minimizing solution. In the same format as that produced by 
                linsolve.LogProductSolver.solve() or linsolve.LinProductSolver.solve().
            wgts: Dictionary that maps equation strings from data to real weights to apply to each 
                equation. Weights are treated as 1/sigma^2. All equations in the data must have a weight 
                if wgts is not the default, {}, which means all 1.0s.
            sparse: Boolean (default False). If True, handles all matrix algebra with sparse matrices. 
                May be faster for certain systems of equations. 
            **kwargs: keyword arguments of constants (python variables in keys of data that 
                are not to be solved for)

        Returns:
            None
        """
        self.prepend = 'd' # XXX make this something hard to collide with
        self.data, self.sparse, self.keys = data, sparse, list(data.keys())
        self.wgts = verify_weights(wgts, self.keys)
        constants = kwargs.pop('constants', kwargs)
        self.init_kwargs, self.sols_kwargs = constants, deepcopy(constants)
        self.sols_kwargs.update(sol0)
        self.all_terms, self.taylors, self.taylor_keys = self.gen_taylors()
        self.build_solver(sol0) 
        self.dtype = self.ls.dtype
    
    def gen_taylors(self, keys=None):
        '''Parses all terms, performs a taylor expansion, and maps equation keys to taylor expansion keys.'''
        if keys is None: keys = self.keys
        all_terms = [ast_getterms(ast.parse(k, mode='eval')) for k in keys]
        taylors, taylor_keys = [], {}
        for terms, k in zip(all_terms, keys):
            taylor = taylor_expand(terms, self.init_kwargs, prepend=self.prepend)
            taylors.append(taylor)
            taylor_keys[k] = jointerms(taylor[len(terms):])
        return all_terms, taylors, taylor_keys

    def build_solver(self, sol0):
        '''Builds a LinearSolver using the taylor expansions and all relevant constants.
        Update it with the latest solutions.'''
        dlin, wlin = {}, {}
        for k in self.keys:
            tk = self.taylor_keys[k]
            dlin[tk] = self.data[k] #in theory, this will always be replaced with data - ans0 before use
            try: wlin[tk] = self.wgts[k]
            except(KeyError): pass
        self.ls = LinearSolver(dlin, wgts=wlin, sparse=self.sparse, constants=self.sols_kwargs)
        self.eq_dict = {eq.val: eq for eq in self.ls.eqs} #maps taylor string expressions to linear equations 
        #Now make sure every taylor equation has every relevant constant, even if they don't appear in the derivative terms.
        for k,terms in zip(self.keys, self.all_terms):
            for term in terms:
                for t in term:
                    t_name = get_name(t)
                    if t_name in self.sols_kwargs:
                        self.eq_dict[self.taylor_keys[k]].add_const(t_name, self.sols_kwargs)
        self._update_solver(sol0)

    def _update_solver(self, sol):
        '''Update all constants in the internal LinearSolver and its LinearEquations based on new solutions.
        Also update the residuals (data - ans0) for next iteration.'''
        self.sol0 = sol
        self.sols_kwargs.update(sol)
        for eq in self.ls.eqs:
            for c in list(eq.consts.values()): 
                if c.name in sol: eq.consts[c.name].val = self.sols_kwargs[c.name]
            self.ls.consts.update(eq.consts)
        ans0 = self._get_ans0(sol)
        for k in ans0: self.ls.data[self.taylor_keys[k]] = self.data[k]-ans0[k]

    def _get_ans0(self, sol, keys=None):
        '''Evaluate the system of equations given input sol. 
        Specify keys to evaluate only a subset of the equations.'''
        if keys is None: 
            keys = self.keys
            all_terms = self.all_terms
            taylors = self.taylors
        else:
            all_terms, taylors, _ = self.gen_taylors(keys)
        ans0 = {}
        for k,taylor,terms in zip(keys,taylors,all_terms):
            eq = self.eq_dict[self.taylor_keys[k]]
            ans0[k] = np.sum([eq.eval_consts(t) for t in taylor[:len(terms)]], axis=0)
        return ans0

    def solve(self, rcond=1e-10, mode='default', verbose=False):
        '''Executes one iteration of a LinearSolver on the taylor-expanded system of 
        equations, improving sol0 to get sol.

        Args:
            rcond: cutoff ratio for singular values useed in numpy.linalg.lstsq, numpy.linalg.pinv,
                or (if sparse) as atol and btol in scipy.sparse.linalg.lsqr
            mode: 'default', 'lsqr', 'pinv', or 'solve', selects which inverter to use. 
                'default': tries 'lsqr' but if a LinAlgError is encountered, backs off to try 'pinv'.
                'lsqr': uses numpy.linalg.lstsq to do an inversion-less solve.  Usually 
                    the fastest solver.
                'solve': uses numpy.linalg.solve to do an inversion-less solve.  Fastest, 
                    but only works for fully constrained systems of equations.
                'pinv': uses numpy.linalg.pinv to perform a pseudo-inverse and then solves.  Can
                    sometimes be more numerically stable (but slower) than 'lsqr'.
                All of these modes are superceded if the same system of equations applies
                to all datapoints in an array.  In this case, a inverse-based method is used so
                that the inverted matrix can be re-used to solve all array indices.
            verbose: print information about iterations

        Returns:
            sol: a dictionary of complex solutions with variables as keys
        '''
        dsol = self.ls.solve(rcond=rcond, mode=mode, verbose=verbose)
        sol = {}
        for dk in dsol:
            k = dk[len(self.prepend):]
            sol[k] = self.sol0[k] + dsol[dk]
        return sol
    
    def eval(self, sol, keys=None):
        '''Returns a dictionary evaluating data keys to the current values given sol and consts.
        Uses the stored data object unless otherwise specified.'''
        if type(keys) is str: keys = [keys]
        elif type(keys) is dict: keys = list(keys.keys())
        return self._get_ans0(sol, keys=keys)
    
    def chisq(self, sol, data=None, wgts=None):
        '''Compute Chi^2 = |obs - mod|^2 / sigma^2 for the specified solution. Weights are treated as 1/sigma^2. 
        wgts = {} means sigma = 1. Uses the stored data and weights unless otherwise overwritten.'''
        if data is None: 
            data = self.data
        if wgts is None: 
            wgts = self.wgts
        wgts = verify_weights(wgts, list(data.keys()))
        return self.ls._chisq(sol, data, wgts, self.eval)

    def solve_iteratively(self, conv_crit=1e-10, maxiter=50, mode='default', verbose=False):
        """Repeatedly solves and updates linsolve until convergence or maxiter is reached. 
        Returns a meta object containing the number of iterations, chisq, and convergence criterion.

        Args:
            conv_crit: A convergence criterion (default 1e-10) below which to stop iterating. 
                Converegence is measured L2-norm of the change in the solution of all the variables
                divided by the L2-norm of the solution itself.
            maxiter: An integer maximum number of iterations to perform before quitting. Default 50.
            mode: 'default', 'lsqr', 'pinv', or 'solve', selects which inverter to use. 
                'default': tries 'lsqr' but if a LinAlgError is encountered, backs off to try 'pinv'.
                'lsqr': uses numpy.linalg.lstsq to do an inversion-less solve.  Usually 
                    the fastest solver.
                'solve': uses numpy.linalg.solve to do an inversion-less solve.  Fastest, 
                    but only works for fully constrained systems of equations.
                'pinv': uses numpy.linalg.pinv to perform a pseudo-inverse and then solves.  Can
                    sometimes be more numerically stable (but slower) than 'lsqr'.
                All of these modes are superceded if the same system of equations applies
                to all datapoints in an array.  In this case, a inverse-based method is used so
                that the inverted matrix can be re-used to solve all array indices.
            verbose: print information about iterations

        Returns: meta, sol
            meta: a dictionary with metadata about the solution, including
                iter: the number of iterations taken to reach convergence (or maxiter)
                chisq: the chi^2 of the solution produced by the final iteration
                conv_crit: the convergence criterion evaluated at the final iteration
            sol: a dictionary of complex solutions with variables as keys
        """
        for i in range(1,maxiter+1):
            if verbose: print('Beginning iteration %d/%d' % (i,maxiter))
            # rcond=conv_crit works because you can't get better precision than the accuracy of your inversion
            # and vice versa, there's no real point in inverting with greater precision than you are shooting for
            new_sol = self.solve(rcond=conv_crit, verbose=verbose, mode=mode)
            deltas = [new_sol[k]-self.sol0[k] for k in new_sol.keys()]
            conv = np.linalg.norm(deltas, axis=0) / np.linalg.norm(list(new_sol.values()),axis=0)
            if np.all(conv < conv_crit) or i == maxiter:
                meta = {'iter': i, 'chisq': self.chisq(new_sol), 'conv_crit': conv}
                return meta, new_sol
            self._update_solver(new_sol)
