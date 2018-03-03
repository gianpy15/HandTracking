import numpy as np
from numpy.linalg import norm
from geometry.transforms import normalize
from scipy.optimize import brentq


def compute_sphere_intersection(center, radius, line, sign=None):
    # <tv-c, tv-c> = r2
    # t2 -2<v, c>t + <c, c> - r2 = 0
    bh = - np.dot(line, center)
    delta = bh ** 2 - np.dot(center, center) + radius ** 2
    if delta < 0:
        if sign is not None:
            return None
        return None, None
    if sign is not None:
        return (sign*np.sqrt(delta) - bh) * line
    return (np.sqrt(delta) - bh) * line, (-bh - np.sqrt(delta)) * line


def sphere_has_intersection(center, radius, line):
    # <tv-c, tv-c> = r2
    # t2 -2<v, c>t + <c, c> - r2 = 0
    bh = - np.dot(line, center)
    delta = bh ** 2 - np.dot(center, center) + radius ** 2
    return delta >= 0


def get_sphere_tangent_lines_coefficient(line1, line2, center, radius):
    # ||tv - C || = r^2
    # has one single solution iff: <v, C>^2 = ||C||^2 - r^2
    # if v = k*v1 + (1-k)*v2 then we need
    # k = (+-sqrt(||C||^2 - r^2) - <v2, C>)/(<v1, C> - <v2, C>)
    # we want also 0 <= k <= 1
    dot1 = np.dot(line1, center)
    dot2 = np.dot(line2, center)
    var_part = np.sqrt(np.dot(center, center) - radius**2)
    k1 = (var_part - dot2) / (dot1 - dot2)
    if 0 <= k1 <= 1:
        return k1
    return (-var_part - dot2) / (dot1 - dot2)


def line_combination(line1, line2, k, dot=None):
    # ||line1 * k + line2 * h|| = 1
    # with 0 <= k <= 1 and 0 <= h <= 1
    # iff h = -k<line1, line2> + sqrt(k^2 <line1, line2>^2 - k^2 +1)

    # optional optimization
    if dot is None:
        dot = np.dot(line1, line2)
    h = -k * dot + np.sqrt(1 + (dot**2 - 1) * k**2)
    return line1 * k + line2 * h


def boundary_distance(k, start_line, reference_line,
                      center, radius,
                      norm_vers, tang_vers,
                      normcos, planecos,
                      sign, refstartdot=None):
    proposed_line = line_combination(start_line, reference_line, k, dot=refstartdot)
    proposed_point = compute_sphere_intersection(center, radius, proposed_line, sign)
    rel_proposed_point = proposed_point - center
    actual_norm_cos = np.dot(rel_proposed_point, norm_vers) / radius

    rel_plane_projection = np.dot(rel_proposed_point, norm_vers) * norm_vers + \
                           np.dot(rel_proposed_point, tang_vers) * tang_vers
    actual_plane_cos = np.dot(rel_proposed_point, rel_plane_projection) / norm(rel_plane_projection) / radius
    norm_cos_diff = actual_norm_cos - normcos
    plane_cos_diff = actual_plane_cos - planecos

    return min(norm_cos_diff, plane_cos_diff)


def find_best_point_in_cone(center, norm_vers, tang_vers, radius, normcos, planecos, objline):

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

    reference_sph_point = center + norm_vers * radius
    reference_versor = normalize(reference_sph_point)

    # reduce the search space by making the objective line nearer
    # in case it is not intersecting the constraint sphere
    if not sphere_has_intersection(center, radius, objline):
        # tang_line = normalize(inner_line * k + outer_line * (1-k)
        k = get_sphere_tangent_lines_coefficient(reference_versor, objline, center, radius)
        start_line = normalize(k * reference_versor + (1-k) * objline)
    else:
        start_line = objline

    # find the interesting side of the sphere
    fp1, fp2 = compute_sphere_intersection(center=center,
                                           radius=radius,
                                           line=reference_versor)
    if norm(fp1-reference_sph_point) < norm(fp2-reference_sph_point):
        sign = +1
    else:
        sign = -1

    refstartdot = np.dot(reference_versor, start_line)
    if boundary_distance(k=1.0,
                         start_line=start_line,
                         reference_line=reference_versor,
                         center=center,
                         radius=radius,
                         norm_vers=norm_vers,
                         tang_vers=tang_vers,
                         normcos=normcos,
                         planecos=planecos,
                         sign=sign,
                         refstartdot=refstartdot) >= 0:
        return compute_sphere_intersection(center, radius, start_line, sign)

    k = brentq(f=boundary_distance,
               a=1e-10,
               b=1.0,
               args=(start_line,
                     reference_versor,
                     center,
                     radius,
                     norm_vers,
                     tang_vers,
                     normcos,
                     planecos,
                     sign,
                     refstartdot))
    return compute_sphere_intersection(center,
                                       radius,
                                       line_combination(start_line,
                                                        reference_versor,
                                                        k,
                                                        refstartdot),
                                       sign)
