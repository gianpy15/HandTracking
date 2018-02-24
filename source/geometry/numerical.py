from scipy.optimize import fsolve, minimize
import numpy as np
from geometry.transforms import *
from numpy.linalg import norm

# debug
from geometry.hand_localization_expl import drawpnts

# BASE PALM POINTS DIRECT INFERENCE

def line_projection_system(sols, params):
    return (sols[0] ** 2 - 2 * params["c01"] * sols[0] * sols[1] + sols[1] ** 2 - params["d01"] ** 2,
            sols[0] ** 2 - 2 * params["c02"] * sols[0] * sols[2] + sols[2] ** 2 - params["d02"] ** 2,
            sols[1] ** 2 - 2 * params["c12"] * sols[1] * sols[2] + sols[2] ** 2 - params["d12"] ** 2)


def line_projection_system_jac(sols, params):
    return [[2 * sols[0] - 2 * params["c01"] * sols[1],
            2 * sols[0] - 2 * params["c02"] * sols[2],
            0],
            [-2 * params["c01"] * sols[0] + 2 * sols[1],
             0,
             2 * sols[1] - 2 * params["c12"] * sols[2]],
            [0,
             -2 * params["c02"] * sols[0] + 2 * sols[2],
             -2 * params["c12"] * sols[1] + 2 * sols[2]]]


def get_points_projections_to_lines(basepts, lines, maxerr=1e-3, maxrestart=1000):
    params = {
        "c01": np.dot(lines[0], lines[1]),
        "c02": np.dot(lines[0], lines[2]),
        "c12": np.dot(lines[1], lines[2]),
        "d01": np.linalg.norm(basepts[0]-basepts[1]),
        "d02": np.linalg.norm(basepts[0]-basepts[2]),
        "d12": np.linalg.norm(basepts[1]-basepts[2])
    }

    dists = np.array([params["d01"], params["d02"], params["d12"]])
    cosines = np.array([params["c01"], params["c02"], params["c12"]])
    # avg = np.average(dists)
    # variance = np.var(dists)
    avg = dists[np.argmax(cosines)]/np.max(cosines)
    variance = np.var(dists / cosines)
    bestsol = None
    besterr = np.inf
    for i in range(maxrestart):
        start = np.random.normal(loc=avg, scale=variance, size=(3,))
        sol, info, stat, msg = fsolve(line_projection_system, start, params, full_output=True)
        err = np.linalg.norm(info["fvec"])
        if err < maxerr:
            if sol[0] < 0:
                sol = -sol
            return np.array([lines[i] * sol[i] for i in range(3)])
        if err < besterr:
            if sol[0] < 0:
                bestsol = -sol
            else:
                bestsol = sol
    print("WARNING: Maxrestarts expired, the proposed solution has error %f" % besterr)
    return np.array([lines[i] * bestsol[i] for i in range(3)])


def get_points_projection_to_lines_pair(basepts, lines, maxerr=1e-3, maxrestart=10, maxtrials=10):
    pts = get_points_projections_to_lines(basepts=basepts, lines=lines)
    pts2 = get_points_projections_to_lines(basepts=basepts, lines=lines)

    count = 1
    while np.linalg.norm(pts - pts2) < 1e-5 and count < maxtrials:
        count += 1
        pts2 = get_points_projections_to_lines(basepts=basepts,
                                               lines=lines,
                                               maxerr=maxerr,
                                               maxrestart=maxrestart)
    return pts, pts2


# CONSTRAINED FINGER INFERENCE

OBJ_LINE = 'v'
CENTER = 'c'
RAD = 'r'
NORM_COS = 'kn'
PLANE_COS = 'kp'
BASEREF = 'br'

numeric_dbg_canvas = None
numeric_dbg_cal = None

def prepare_problem_params(center, norm_v, tang_v, radius, normcos, planecos, objline):
    # perform a base change -> norm as x-axis, tang as y-axis
    # to optimize the objective evaluation
    conorm = cross_product(norm_v, tang_v)
    # drawpnts([objline], fill="magenta", canvas=numeric_dbg_canvas, cal=numeric_dbg_cal)
    params = {
        BASEREF: np.stack((norm_v, tang_v, conorm)),
        RAD: radius,
        NORM_COS: normcos,
        PLANE_COS: planecos
    }
    params[CENTER] = column_matmul(params[BASEREF], center)
    params[OBJ_LINE] = column_matmul(params[BASEREF], objline)
    # drawpnts([column_matmul(np.transpose(params[BASEREF]), params[OBJ_LINE])], fill="grey", canvas=numeric_dbg_canvas, cal=numeric_dbg_cal)
    return params


def build_constraints(params: dict):
    return ({'type': 'ineq', 'fun': subject_to_cosplane, 'args': (params[PLANE_COS] ** 2,)},
            {'type': 'ineq', 'fun': subject_to_existence})


def build_bounds(params: dict):
    an_min = params[NORM_COS]
    an_max = 1.0
    at_max = np.sqrt(1.0 - params[NORM_COS] ** 2)
    return (an_min, an_max), (-at_max, at_max)


def trd_sol_elem(proposed_sol):
    arg = 1 - proposed_sol[0]**2 - proposed_sol[1]**2
    if arg <= 0:
        return 0, 0
    ret = np.sqrt(arg)
    return -ret, ret


def rel_pnt(proposed_sol: np.ndarray, rad):
    p1, p2 = trd_sol_elem(proposed_sol)
    return (np.array([proposed_sol[0], proposed_sol[1], p1])*rad,
            np.array([proposed_sol[0], proposed_sol[1], p2])*rad)


def take_final_sol(proposed_sol: np.ndarray, params: dict):
    fp1, fp2 = rel_pnt(proposed_sol, params[RAD]) + params[CENTER]
    ret1 = - np.dot(fp1, params[OBJ_LINE]) / norm(fp1)
    ret2 = - np.dot(fp2, params[OBJ_LINE]) / norm(fp2)
    if ret1 < ret2:
        return fp1
    return fp2

# lastloss = None
# lastcol = 128


def minimizing_obj(proposed_sol: np.ndarray, params: dict):
    fp1, fp2 = rel_pnt(proposed_sol, params[RAD]) + params[CENTER]
    ret1 = - np.dot(fp1, params[OBJ_LINE]) / norm(fp1)
    ret2 = - np.dot(fp2, params[OBJ_LINE]) / norm(fp2)
    loss = min(ret1, ret2)
    # DEBUG CODE
    # global lastloss
    # global lastcol
    # if lastloss is not None:
    #     col = lastcol + (lastloss - loss)/0.01 * 5
    #     if col < 0:
    #         col = 0
    #     lastcol = col
    # else:
    #     lastcol = 128
    #     col = 128
    #     lastloss = loss
    # drawpnts([column_matmul(np.transpose(params[BASEREF]), fp1)],
    #          canvas=numeric_dbg_canvas,
    #          cal=numeric_dbg_cal,
    #          fill="#%02X%02X%02X"%(int(col), int(col), int(col)))

    # print("Loss: %f" % min(ret1, ret2))
    # import time
    # time.sleep(0.2)
    return loss


def subject_to_cosplane(proposed_sol: np.ndarray, spcos):
    return norm(proposed_sol) ** 2 - spcos


def subject_to_existence(proposed_sol: np.ndarray):
    return 1 - norm(proposed_sol) ** 2


def extract_solution(proposed_sol: np.ndarray, params: dict):
    pnt = take_final_sol(proposed_sol, params)
    ret = column_matmul(np.transpose(params[BASEREF]), pnt)
    return ret


def find_best_point_in_cone(center, norm_vers, tang_vers, radius, normcos, planecos, objline, dbgcanvas, dbgcal):

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
        tang_vers = cross_product(cross_product(norm_vers, tang_vers), norm_vers)

    tang_vers = checknorm(tang_vers)
    objline = checknorm(objline)

    params = prepare_problem_params(center, norm_vers, tang_vers, radius, normcos, planecos, objline)
    bounds = build_bounds(params)
    constr = build_constraints(params)

    def action():
        global res
        res = minimize(minimizing_obj, np.array([1.0, 0.0]), args=params, bounds=bounds, constraints=constr)

    # import timeit
    # print("Optimization problem solved in %f ms." % (1000 * timeit.timeit(action, number=1),))
    action()
    if not res.success:
        print("Optimization failure. Message: %s" % res.message)
        print(res.x)

    # print("COSPLANE CONSTR: %f" % subject_to_cosplane(proposed_sol=res.x, spcos=params[PLANE_COS]**2))
    # print("EXISTENCE CONSTR: %f" % subject_to_existence(proposed_sol=res.x))
    # print("OBJECTIVE: %f" % minimizing_obj(proposed_sol=res.x, params=params))
    # import time
    # time.sleep(2)
    # numeric_dbg_canvas.delete("debug")
    return extract_solution(res.x, params)
