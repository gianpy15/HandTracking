from numba import jit
import numpy as np
import tqdm
import data.datasets.crop.utils as u
from data.naming import *
import matplotlib.pyplot as mplt


def loop_pix_avg_dist(yt, yp):
    thrs = np.linspace(0, 1, 11)
    dist = []
    std = []
    for thr in thrs:
        d, s = pixel_avg_dist_euclidean(yt, yp, thr, False)
        dist.append(d)
        std.append(s)
    print("###########")
    print(thrs)
    print(dist)
    print(std)
    plot(thrs, dist, "thresholds", "mean distance", "1.jpg")
    plot(thrs, std, "thresholds", "mean std", "2.jpg")


def loop_prec_recall(yt, yp):
    thrs = np.linspace(0, 1, 11)
    prec = []
    reca = []
    for thr in thrs:
        prec.append(precision(yt, yp, thr, False))
        reca.append(recall(yt, yp, thr, False))
    print("###########")
    print("THRS: ", thrs)
    print("PREC: ", prec)
    print("RECA: ", reca)
    plot(thrs, prec, "thresholds", "precision", "3.jpg")
    plot(thrs, reca, "thresholds", "recall", "4.jpg")


@jit
def precision(yt, yp, thr=0.5, verb=True):
    yt = np.array(yt)
    yp = np.array(yp)
    ris = []
    for i in tqdm.tqdm(range(len(yt))):
        ris.append(__precision(yp[i], yt[i], thr))
    ris = np.array(ris)
    if verb:
        print("##############")
        print("PRECISION: ", ris.mean())
    return ris.mean()


@jit
def __precision(h1, h2, thr):
    h1 = 1 - (1 - h1)**5
    h1[h1 >= thr] = 1
    h2[h2 >= thr] = 1
    h1[h1 < thr] = 0
    h2[h2 < thr] = 0
    if np.sum(h1) == 0:
        return 0
    return np.sum(h1*h2)/np.sum(h1)


@jit
def recall(yt, yp, thr=0.5, verb=True):
    yt = np.array(yt)
    yp = np.array(yp)
    ris = []
    for i in tqdm.tqdm(range(len(yt))):
        ris.append(__recall(yp[i], yt[i], thr))
    ris = np.array(ris)
    if verb:
        print("##############")
        print("RECALL: ", ris.mean())
    return ris.mean()


@jit
def __recall(h1, h2, thr):
    h1 = 1 - (1 - h1) ** 5
    h1[h1 >= thr] = 1
    h2[h2 >= thr] = 1
    h1[h1 < thr] = 0
    h2[h2 < thr] = 0
    return np.sum(h1*h2)/np.sum(h2)


def pixel_avg_dist_euclidean(yt, yp, thr=0.5, verb=True):
    yp = np.array(yp)
    dists = []
    for i in tqdm.tqdm(range(len(yt))):
        try:
            yp[i][yp[i] >= thr] = 1
            tcx, tcy = __heatmap_centroid(yt[i])
            pcx, pcy = __heatmap_centroid(yp[i])
            dist = np.sqrt(((tcx - pcx)**2) + ((tcy - pcy)**2))
            dists.append(dist)
        except Exception:
            continue
    dists = np.array(dists)
    if verb:
        print("##############")
        print("MEAN CENTROIDS PIXEL DISTANCE: ", dists.mean())
        print("STD CENTROIDS PIXEL DISTANCE: ", dists.std())
    return dists.mean(), dists.std()


def percentage_overlap(yt, yp, thr=0.5):
    yp = np.array(yp)
    yt = np.array(yt)
    yp = np.array(yp)
    print(yt.shape, yp.shape)
    ris = []
    for i in tqdm.tqdm(range(len(yt))):
        ris.append(__percentage_overlap(yp[i], yt[i], thr))
    ris = np.array(ris)
    print("##############")
    print("PERC OVERLAP: ", ris.mean())


@jit
def __percentage_overlap(h1, h2, thr):
    h1[h1 > thr] = 1
    return np.sum(h1*h2)/np.sum(h2)


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


def create_sprite(n, f, h, num=1):
    path = resources_path(os.path.join("saves_for_report", "sprite"))
    os.makedirs(path, exist_ok=True)
    sprite = None
    for i in range(n):
        row = None
        for j in range(n):
            pos = np.random.randint(0, len(f))
            f_ts = f[pos]
            h_ts = h[pos]
            heat3d = np.dstack((h_ts, h_ts, h_ts))
            heat3d[heat3d < 0.3] = 0.3
            if row is None:
                row = np.uint8(heat3d * f_ts)
            else:
                row = np.array(np.hstack((row, np.uint8(heat3d * f_ts))), dtype=np.uint8)
        if sprite is None:
            sprite = row
        else:
            sprite = np.vstack((sprite, row))
    u.save_image(os.path.join(path, str(num) + ".jpg"), sprite)


def print_one_pred(yt, yp, num=0):
    u.showimage(yt[num])
    u.showimage(yp[num])


def plot(x, y, xlab, ylab, name=None):
    mplt.plot(x, y)
    mplt.xlabel(xlab)
    mplt.ylabel(ylab)
    if name is not None:
        mplt.savefig(resources_path("graphs", name))
    mplt.show()
