import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import minimize, rosen, rosen_der
from mpl_toolkits.mplot3d.axes3d import Axes3D
import math
import image_util as iu
import time

# grandezza degli eventi da considerare ad ogni iterazione per stimare i parametri theta
Ne = 80000


def choose_starting_point(chunk, v_min, v_max, step, events_batch):
    s_tx = s_ty = 0
    best_var = 0

    for x in range(v_min, v_max, step):
        iwe = iu.warped_event(chunk, chunk.iloc[0, 0], x, 0, events_batch)
        var = np.var(iwe)

        if var > best_var:
            best_var = var
            s_tx = x

    for y in range(v_min, v_max, step):
        iwe = iu.warped_event(chunk, chunk.iloc[0, 0], s_tx, y, events_batch)
        var = np.var(iwe)

        if var > best_var:
            best_var = var
            s_ty = y

    print(f"Starting point: {s_tx} e {s_ty}")

    return s_tx, s_ty


def bruteforce_optimization(vector, tref, v_min_x, v_max_x, v_min_y, v_max_y, step, events_batch):
    graph = pd.DataFrame(columns=['X', 'Y', 'VAR'])

    best = 0
    best_vx = 0
    best_vy = 0
    for vy in range(v_min_y, v_max_y, step):
        for vx in range(v_min_x, v_max_x, step):
            iwe = iu.warped_event(vector, tref, vx, vy, events_batch)
            # val = cv2.meanStdDev(iwe)
            # value = val[1]
            value = np.var(iwe)

            graph = graph.append({'X': vx, 'Y': vy, 'VAR': value}, ignore_index=True)

            #graph.append({'X': vx, 'Y': vy, 'VAR': value}, ignore_index=True)

            if value > best:
                best = value
                best_vy = vy
                best_vx = vx

    # print("end-- bruteforce optimization -- best value: " + str(best) + " vx: " + str(best_vx) + " vy: " + str(best_vy))
    return best_vx, best_vy, graph


def indipendent_optimization(chunk, step, events_batch):
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)

    best_iwe = np.zeros((180, 240), dtype=np.uint8)

    start = time.time()
    init_tx, init_ty = choose_starting_point(chunk, -250, 250, 30, events_batch)
    end = time.time()

    print(f"Choosen starting point {init_tx} and {init_ty} in {end - start} s")

    s_tx = s_ty = 0

    best_var = 0

    start = time.time()

    for x in range(-25, 25, step):
        tx = init_tx + x
        ty = init_ty

        iwe = iu.warped_event(chunk, chunk.iloc[0, 0], tx, ty, events_batch)
        var = cv2.meanStdDev(iwe)[1] ** 2

        if var > best_var:
            best_var = var
            s_tx = tx
            best_iwe = iwe

    for y in range(-25, 25, step):
        tx = s_tx
        ty = init_ty + y

        iwe = iu.warped_event(chunk, chunk.iloc[0, 0], tx, ty, events_batch)
        var = cv2.meanStdDev(iwe)[1] ** 2

        if var > best_var:
            best_var = var
            s_ty = ty
            best_iwe = iwe

    end = time.time()

    """
    cv2.namedWindow("Velocity", cv2.WINDOW_AUTOSIZE)
    black = np.zeros((100, 200), dtype=np.uint8)

    cv2.putText(black, f"Time: {chunk.iloc[0, 0]}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv2.putText(black, f"VX: {s_tx} px/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv2.putText(black, f"VY: {s_ty} px/s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    cv2.imshow("Velocity", black)
    cv2.waitKey(1)

    cv2.imshow("IWE", best_iwe * 10)
    cv2.waitKey(1)

    iu.difference_image(chunk, 3500)

    print(f"Parameter after search - VX {s_tx} and VY {s_ty} in {end - start} s")
    """

    return s_tx, s_ty


def search_gradient(vector, tref, startvx, startvy):
    best_vx = startvx
    best_vy = startvy
    best_var = 0
    power = [10, 5, 1]
    #directions = [(0, 0), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    directions = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]

    update = 1;
    while update:
        update = 0
        # find best direction
        dvx = 0
        dvy = 0
        dvar = 0

        old_iwe = iu.warped_event(vector, tref, best_vx, best_vy, 100)
        # resize image
        start_var = np.var(old_iwe)

        for p in power:
            for e in directions:
                # print(e)
                new_vx = best_vx + e[0] * p
                new_vy = best_vy + e[1] * p
                new_iwe = iu.warped_event(vector, tref, new_vx, new_vy, 100)
                new_var = np.var(new_iwe)
                if new_var - start_var > dvar:
                    update = 1
                    dvar = new_var - start_var
                    dvx = e[0] * p
                    dvy = e[1] * p
                    print("UPDATE")
                '''
                print(" direction: x: " + str(e[0]) + " y: " + str(e[1]) + " value: " + str(new_var) + " vx: " + str(
                    new_vx) + " vy: " + str(new_vy) + " best value: " + str(
                    best_var) + " With best vx: " + str(best_vx) + ", vy: " + str(best_vy))
                '''

        best_vx = best_vx + dvx
        best_vy = best_vy + dvy
        best_var = best_var + dvar

        print(" direction: x: " + str(e[0]) + " y: " + str(e[1]) + " value: " + str(new_var) + " vx: " + str(
            new_vx) + " vy: " + str(new_vy) + " best value: " + str(
            best_var) + " With best vx: " + str(best_vx) + ", vy: " + str(best_vy))

    return best_vx, best_vy


def main():
    df_chunk = pd.read_csv(r'outdoors_walking/events.txt', chunksize=Ne, sep="\s+", names=["timestamp", "x", "y", "p"])
    data = pd.DataFrame(columns=["alg", "vx", "vy", "error", "time"])

    i = 1
    a = 0

    bruteforce_x = [-56, -68, -104, -88, -50, -54, -124, -100]
    bruteforce_x = [-14, 32, 6, -14, -4, 58, -38, 66]

    for chunk in df_chunk:
        pos = 2000000

        if 4 <= i <= 18 and i % 2 == 0:
            vector = chunk
            tref = chunk.iloc[0, 0]

            # generate_var_image(chunk.iloc[pos:pos + Ne, :])

            print("tref: ", tref)

            # Brute Force
            start = time.time()
            x, y, graph = bruteforce_optimization(vector, tref, -250, 250, -250, 250, 2, 100)
            end = time.time()

            data = data.append({"alg": "brut", "vx": x, "vy": y, "error": 0, "time": end-start}, ignore_index=True)

            print(f"Bruteforce vx: {x} vy: {y} time: {end-start}")

            opt_x = x
            opt_y = y

            # Starting point + Brute Force
            start = time.time()
            sx, sy = choose_starting_point(vector, -250, 250, 30, 100)
            x, y, graph = bruteforce_optimization(vector, tref, sx - 25, sx + 25, sy - 25, sy + 25, 2, 100)
            end = time.time()

            error = math.sqrt((opt_x - x) ** 2 + (opt_y - y) ** 2)
            data = data.append({"alg": "startbrut", "vx": x, "vy": y, "error": error, "time": end-start}, ignore_index=True)

            print(f"Starting + Bruteforce vx: {x} vy: {y} time: {end-start}")

            # Starting point + Indipendent Optimization
            start = time.time()
            x, y = indipendent_optimization(vector, 2, 70)
            end = time.time()

            error = math.sqrt((opt_x - x) ** 2 + (opt_y - y) ** 2)
            data = data.append({"alg": "indipendent", "vx": x, "vy": y, "error": error, "time": end - start}, ignore_index=True)

            print(f"Indipendent vx: {x} vy: {y} time: {end - start}")

            # Starting point + Indipendent Optimization
            start = time.time()
            sx, sy = choose_starting_point(vector, -250, 250, 30, 100)
            x, y = search_gradient(vector, tref, sx, sy)
            end = time.time()

            error = math.sqrt((opt_x - x) ** 2 + (opt_y - y) ** 2)
            data = data.append({"alg": "gradient", "vx": x, "vy": y, "error": error, "time": end - start}, ignore_index=True)

            print(f"Gradient vx: {x} vy: {y} time: {end - start}")

            print(data)
            data.to_csv('data.csv', mode='a', index=False)

            a += 1

        elif i > 20:
            break

        i += 1


if __name__ == "__main__":
    main()
