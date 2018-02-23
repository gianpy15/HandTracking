import numpy as np
from geometry.transforms import *
from geometry.calibration import *
import timeit


def inner_angle(v1, v2):
    return np.arccos(np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2))


def projections_benchmark(repeat=1000):
    times = []
    good_times = []
    cal = calibration(synth_intrinsic((500, 500), (50, 50)))
    p1 = np.array(np.random.uniform(low=-5, high=5.0, size=(3,)))
    p2 = np.array(np.random.uniform(low=-5, high=5.0, size=(3,)))
    p3 = np.array(np.random.uniform(low=-5, high=5.0, size=(3,)))
    for i in range(repeat):
        l1 = np.array(np.random.uniform(low=0, high=500, size=(2,)))
        l2 = np.array(np.random.uniform(low=0, high=500, size=(2,)))
        l3 = np.array(np.random.uniform(low=0, high=500, size=(2,)))

        l1 = ImagePoint(l1).to_camera_model(cal).as_row()
        l2 = ImagePoint(l2).to_camera_model(cal).as_row()
        l3 = ImagePoint(l3).to_camera_model(cal).as_row()

        p1 = np.array(np.random.uniform(low=-5, high=5.0, size=(3,)))
        p2 = np.array(np.random.uniform(low=-5, high=5.0, size=(3,)))
        p3 = np.array(np.random.uniform(low=-5, high=5.0, size=(3,)))

        lines = np.array([l1, l2, l3])

        basepts = np.array([p1, p2, p3])

        time = timeit.timeit(lambda: get_points_projection_to_lines_pair(basepts, lines), number=1)
        times.append(time)
        if time > 0.5:
            print("Execution stuck for %f seconds on a triangle with angles:" % time)
            print(inner_angle(p1 - p2, p3 - p2) / np.pi * 180)
            print(inner_angle(p2 - p1, p3 - p1) / np.pi * 180)
            print(inner_angle(p1 - p3, p2 - p3) / np.pi * 180)
        else:
            good_times.append(time)
    return times, good_times


if __name__ == '__main__':
    tms, gtms = projections_benchmark(100)
    print("-------------------------------------")
    print("Results:")
    print("Average call execution time: %f" % np.average(tms))
    print("Maximum call execution time: %f" % np.max(tms))
    print("Minimum call execution time: %f" % np.min(tms))
    print("Call execution time variance: %f" % np.var(tms))
    print("Total calls: %d" % len(tms))
    print("-------------------------------------")
    print("Except ill cases:")
    print("Average call execution time: %f" % np.average(gtms))
    print("Maximum call execution time: %f" % np.max(gtms))
    print("Minimum call execution time: %f" % np.min(gtms))
    print("Call execution time variance: %f" % np.var(gtms))
    print("Total calls: %d" % len(gtms))

