from scipy.optimize import minimize
from library.geometry.transforms import *
from numpy.linalg import norm

# CONSTRAINED FINGER INFERENCE

OBJ_LINE = 'v'
CENTER = 'c'
RAD = 'r'
NORM_COS = 'kn'
PLANE_COS = 'kp'
BASEREF = 'br'
SIGN = 's'

numeric_dbg_canvas = None
numeric_dbg_cal = None


def prepare_problem_params(center, norm_v, tang_v, radius, normcos, objline):
    # perform a base change -> norm as x-axis, tang as y-axis
    # to optimize the objective evaluation
    conorm = np.cross(norm_v, tang_v)
    params = {
        BASEREF: np.stack((norm_v, tang_v, conorm)),
        RAD: radius,
        NORM_COS: normcos,
        SIGN: None
    }
    params[CENTER] = params[BASEREF] @ center
    params[OBJ_LINE] = params[BASEREF] @ objline
    return params


def build_bounds(params: dict):
    an_min = params[NORM_COS]
    an_max = 1.0
    return (an_min, an_max),


def truncate_by_bounds(proposed_sol, bounds):
    out = proposed_sol[0]
    if out < bounds[0][0]:
        out = bounds[0][0]
    elif out > bounds[0][1]:
        out = bounds[0][1]
    return np.array([out])


def snd_sol_elem(proposed_sol, sign):
    arg = 1 - proposed_sol[0]**2
    if sign is not None:
        if arg <= 0:
            return np.zeros(shape=(1,))
        return np.array([sign * np.sqrt(arg)])
    if arg <= 0:
        return np.zeros(shape=(2, 1))
    ret = np.sqrt(arg)
    return np.array([[-ret], [ret]])


def rel_pnt(proposed_sol, rad, sign):
    if sign is None:
        return np.concatenate((np.tile(proposed_sol, (2, 1)),
                               snd_sol_elem(proposed_sol, sign),
                               np.zeros(shape=(2, 1))), axis=1)*rad
    return np.concatenate((proposed_sol, snd_sol_elem(proposed_sol, sign), [0]))*rad


def take_final_sol(proposed_sol: np.ndarray, params: dict):
    if params[SIGN] is None:
        fp = rel_pnt(proposed_sol, params[RAD], sign=None) + params[CENTER]
        ret = -np.dot(fp, params[OBJ_LINE]) / norm(fp, axis=1)
        if ret[0] < ret[1]:
            return fp[0]
        return fp[1]
    return rel_pnt(proposed_sol, params[RAD], sign=params[SIGN]) + params[CENTER]


# lastloss = None
# lastcol = 128


def minimizing_obj(proposed_sol: np.ndarray, params: dict):
    if params[SIGN] is None:
        fp = rel_pnt(proposed_sol, params[RAD], sign=None) + params[CENTER]
        ret = -np.dot(fp, params[OBJ_LINE]) / norm(fp, axis=1)
        if np.argmin(ret) == 0:
            params[SIGN] = -1
        else:
            params[SIGN] = 1
        return min(ret)
    fp = rel_pnt(proposed_sol, params[RAD], sign=params[SIGN]) + params[CENTER]
    return -np.dot(fp, params[OBJ_LINE]) * 1e+10 / norm(fp)


def extract_solution(proposed_sol: np.ndarray, params: dict):
    pnt = take_final_sol(proposed_sol, params)
    return np.transpose(params[BASEREF]) @ pnt


def find_best_point_in_cone(center, norm_vers, tang_vers, radius, normcos, objline, suggestion=None):

    # global numeric_dbg_canvas
    # numeric_dbg_canvas = dbgcanvas
    # global numeric_dbg_cal
    # numeric_dbg_cal = dbgcal
    # global lastloss
    # lastloss = None

    def checknorm(subj):
        nrm = norm(subj)
        if nrm < 0.999 or nrm > 1.001:
            return subj / nrm
        return subj

    norm_vers = checknorm(norm_vers)

    if np.dot(norm_vers, tang_vers) > 1e-8:
        tang_vers = np.cross(np.cross(norm_vers, tang_vers), norm_vers)

    tang_vers = checknorm(tang_vers)
    objline = checknorm(objline)

    params = prepare_problem_params(center, norm_vers, tang_vers, radius, normcos, objline)
    bounds = build_bounds(params)

    starting_sol = np.ones(shape=(1,))
    idx = 0
    if suggestion is not None:
        bestobj = minimizing_obj(starting_sol, params)
        for sugg in suggestion:
            suggested_vers = normalize(sugg-center)
            base_wise_sugg = [(params[BASEREF] @ suggested_vers)[0]]
            base_wise_sugg = truncate_by_bounds(base_wise_sugg, bounds)
            currobj = minimizing_obj(base_wise_sugg, params)
            idx += 1
            if bestobj > currobj:
                # print("Sugg. %d is better than previous. Taking it" % idx)
                starting_sol = base_wise_sugg
                bestobj = currobj

    def action():
        global res
        res = minimize(minimizing_obj, starting_sol,
                       options={'eps': 1e-20},
                       bounds=bounds,
                       args=params)

    # import timeit
    # print("Optimization problem solved in %f ms." % (1000 * timeit.timeit(action, number=1),))
    action()
    if not res.success:
        print("Optimization failure in arc sector search. Message: %s" % res.message)
        print(res.x)
        return None

    # print("COSPLANE CONSTR: %f" % subject_to_cosplane(proposed_sol=res.x, spcos=params[PLANE_COS]**2))
    # print("EXISTENCE CONSTR: %f" % subject_to_existence(proposed_sol=res.x))
    # print("OBJECTIVE: %f" % minimizing_obj(proposed_sol=res.x, params=params))
    # import time
    # time.sleep(2)
    # numeric_dbg_canvas.delete("debug")
    return extract_solution(res.x, params)
