import pandas as pd
import cv2
import numpy as np
import time

Ne = 50   #grandezza degli eventi da considerare ad ogni iterazione per stimare i parametri theta

def dirac_delta(x, y, x_warped, y_warped):
	delta = (x - x_warped) + (y - y_warped)
	if delta == 0:
		return 1
	else:
		return 0


def generateIWE(warped_event):
    iwe = np.zeros((180, 240), dtype=np.int32)

    # Per ogni posizione dell'immagine
    for i in range(180):
	    for j in range(240):
		    point = 0

		    # Per ogni evento
		    for a in range(len(warped_event)):
			    point += 1 * dirac_delta(j, warped_event.iloc[a, 2], i, warped_event.iloc[a, 1])

		    iwe[i,j] = point
    return iwe

def warped_event(events, tref, theta_x,theta_y):
    worped_e = pd.DataFrame(columns=['time', 'x', 'y'])
    for i in range(len(events)):
        x = int((events.iloc[i, 1])+(events.iloc[i, 0]-tref)*theta_x)
        y = int((events.iloc[i, 2])+(events.iloc[i, 0]-tref)*theta_y)
        worped_e = worped_e.append({'time': tref, 'x': x, 'y': y}, ignore_index=True)

    return worped_e


    # grandezza totale immagine 180V x 240H
def events_to_image(event_list, dt):    # tutti gli eventi all'interno di dt vengono proiettati sull'immagine.... in questo modo sono visibili più punti....
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    i = 0
    image = np.zeros(shape=[180, 240, 1], dtype=np.uint8)
    while i < len(event_list):
        image = np.zeros(shape=[180, 240, 1], dtype=np.uint8)

        for j in range(dt):
            # print(event_list.iloc[i,"x"])
            # image[event_list.iloc[i+j,2],event_list.iloc[i+j,1]] = 0.20 * image[event_list.iloc[i+j,2],event_list.iloc[i+j,1]] * 1/(event_list.iloc[i+j,0] ** 2) + 0.80 * int(event_list.iloc[i+j,3])*255
            image[event_list.iloc[i + j, 2], event_list.iloc[i + j, 1]] = 255

        cv2.waitKey(1)
        cv2.imshow("image", image)
        i = i + dt

# data = pd.read_csv('outdoors_walking/events.txt', header = None)
df_chunk = pd.read_csv(r'outdoors_walking/events.txt', chunksize=4000000, sep="\s+", names=["timestamp", "x", "y", "p"])
for chunk in df_chunk:
    #events_to_image(chunk.iloc[0:4000000, :], 2500)
    w_e = warped_event(chunk.iloc[0:Ne, :],0,0,0)
    iwe = generateIWE(w_e)
    print (iwe)
    break
