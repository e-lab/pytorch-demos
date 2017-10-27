import numpy as np

def nms(boxes, thresh):

    if len(boxes) < 2: return boxes

    boxes = np.array(boxes)
    nboxes = np.array(boxes)
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s  = boxes[:,4]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(s)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
	    # the bounding box and the smallest (x, y) coordinates
	    # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

	    # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

	    # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

	    # delete all indexes from the index list that have
        todel = np.concatenate(([last], np.where(overlap > thresh)[0]))
        idxs = np.delete(idxs, todel)

        if np.any(np.where(overlap > thresh)[0]):
            idxs = np.append(idxs, i)

    return nboxes[pick]
