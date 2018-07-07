from numba import jit
import numpy as np
import tqdm
import data.datasets.crop.utils as u

def pixel_avg_dist_euclidean(yt, yp):
    dists = []
    for i in tqdm.tqdm(range(len(yt))):
        try:
            tcx, tcy = __heatmap_centroid(yt[i])
            pcx, pcy = __heatmap_centroid(yp[i])
            dist = np.sqrt(((tcx - pcx)**2) + ((tcy - pcy)**2))
            print(dist)
            dists.append(dist)
        except Exception:
            continue
    dists = np.array(dists)
    print("##############")
    print("MEAN CENTROIDS PIXEL DISTANCE: ", dists.mean())
    print("STD CENTROIDS PIXEL DISTANCE: ", dists.std())


@jit
def __heatmap_centroid(heat):
    heat = heat.squeeze()
    acc_x = 0
    acc_y = 0
    n = 0
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            acc_x += i * heat[i][j]
            acc_y += j * heat[i][j]
            n += heat[i][j]
    if n == 0:
        raise Exception
    return acc_x/n, acc_y/n


def print_one_pred(yt, yp, num=0):
    u.showimage(yt[num])
    u.showimage(yp[num])
