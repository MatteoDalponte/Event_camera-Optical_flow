import cv2
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np


def difference_image(event_list, dt):
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    i = 0

    start_end_picture = np.zeros(shape=[180, 240, 3], dtype=np.uint8)

    # start image
    for j in range(dt):
        start_end_picture[event_list.iloc[j, 2], event_list.iloc[j, 1]] = (0, 200, 0)

    # end image
    for j in range(len(event_list) - dt, len(event_list) - 1):
        start_end_picture[event_list.iloc[j, 2], event_list.iloc[j, 1]] = (255, 20, 255)

    start_time = event_list.iloc[0, 0]
    end_time = event_list.iloc[len(event_list)-1, 0]

    scaled = np.zeros((360, 480), dtype=np.uint8)
    scaled = cv2.resize(start_end_picture, (480, 360), interpolation=cv2.INTER_CUBIC)

    cv2.putText(scaled, "start: "+str(start_time)+" end: "+str(end_time), (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    cv2.imshow("image", scaled)
    cv2.imwrite("start_end_pic.jpg", start_end_picture)
    cv2.imshow("image1", start_end_picture)

    cv2.waitKey(1)


def events_to_image(event_list, dt):
    # tutti gli eventi all'interno di dt vengono proiettati sull'immagine.... in questo modo sono visibili più punti....
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    i = 0
    image = np.zeros(shape=[180, 240, 1], dtype=np.uint8)
    while i < len(event_list) - dt:
        image = np.zeros(shape=[180, 240, 1], dtype=np.uint8)

        for j in range(dt):
            image[event_list.iloc[i + j, 2], event_list.iloc[i + j, 1]] = 255

        cv2.waitKey(1)
        cv2.imshow("image", image)
        cv2.waitKey(1)
        i = i + dt


def generate_var_image(chunk, batch, step):
    start = time.time()
    M = pd.DataFrame(columns=["x", "y", "z"])
    # var_image = np.zeros((280, 280))

    for x in range(-200, 200, step):
        for y in range(-200, 200, step):
            iwe = warped_event(chunk, chunk.iloc[0, 0], x, y, batch)
            var = cv2.meanStdDev(iwe)[1] ** 2

            M = M.append([{"x": x, "y": y, "z": var[0][0]}], ignore_index=True)

    # generate data points and compute the surface of the function
    Z = M.z

    end = time.time()

    print(f"Image generated: {end - start}")

    # create the figure
    fig = plt.figure(figsize=(15, 10))

    # create a 3D axes object for the first plot, and plot the surface computed before
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('vx (px/s)')
    ax.set_ylabel('vy (px/s)')
    ax.set_zlabel('Variance')
    ax.set_title('Variance distribution')
    p = ax.plot_trisurf(M.x, M.y, Z, linewidth=1, cmap=plt.cm.coolwarm, alpha=0.9)
    cb = fig.colorbar(p, shrink=0.7)

    plt.show()


def warped_event(events, tref, theta_x, theta_y, sample):
    iwe = np.zeros((180, 240), dtype=np.uint8)

    for i in range(0, len(events), sample):
        x = int((events.iloc[i, 1]) - (events.iloc[i, 0] - tref) * theta_x)
        y = int((events.iloc[i, 2]) - (events.iloc[i, 0] - tref) * theta_y)

        if 0 <= y < 180 and 0 <= x < 240:
            iwe[y, x] += 50

    return cv2.blur(iwe, (5, 5))
