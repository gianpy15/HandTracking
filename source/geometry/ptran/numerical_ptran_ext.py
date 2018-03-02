import numpy as np


#pythran export truncate_by_bounds(float[], float[][])
def truncate_by_bounds(proposed_sol, bounds):
    out = proposed_sol[:]
    for i in range(2):
        if out[i] < bounds[i][0]:
            out[i] = bounds[i][0]
        elif out[i] > bounds[i][1]:
            out[i] = bounds[i][1]
    return out


#pythran export trd_sol_elem(float[])
def trd_sol_elem(proposed_sol):
    arg = 1 - proposed_sol[0]**2 - proposed_sol[1]**2
    if arg <= 0:
        return np.zeros(shape=(2, 1))
    ret = np.sqrt(arg)
    return np.array([[-ret], [ret]])


#pythran export rel_pnt(float[], float)
def rel_pnt(proposed_sol, rad):
    return np.concatenate((np.tile(proposed_sol, (2, 1)),
                           trd_sol_elem(proposed_sol)), axis=1)*rad


#pythran export take_final_sol(float[], str:float[] dict)
def take_final_sol(proposed_sol, params):
    fp = rel_pnt(proposed_sol, params['r'][0]) + params['c']
    ret = -np.dot(fp, params['v']) / np.linalg.norm(fp, axis=1)
    if ret[0] < ret[1]:
        return fp[0]
    return fp[1]


#pythran export minimizing_obj(float[], str:float[] dict)
def minimizing_obj(proposed_sol, params):
    fp = rel_pnt(proposed_sol, params['r'][0]) + params['c']
    ret = -np.dot(fp, params['v']) / np.linalg.norm(fp, axis=1)
    return min(ret)
