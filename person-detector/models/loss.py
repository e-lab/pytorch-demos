import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable as V

def detection_loss(pred_conf, pred_bbox, gt):

    p_loss = []
    b_loss = []
    for b in range(len(gt)):

        gt_class = torch.zeros(1)
        gt_bbox  = torch.zeros(1, 4)

        if len(gt[b]) == 0:
            # no box. just bg. bbox loss is useless
            prob_loss = F.cross_entropy(pred_conf[b].unsqueeze(0),
                                        V(gt_class.long()))
            p_loss.append(prob_loss)
            continue

        old_loss = None
        for (x,y,w,h), c in gt[b]:
            gt_bbox[0,0] = x
            gt_bbox[0,1] = y
            gt_bbox[0,2] = w
            gt_bbox[0,3] = h

            gt_class[0] = c

            prob_loss = F.cross_entropy(pred_conf[b].unsqueeze(0),
                                        V(gt_class.long()))
            bbox_loss = F.mse_loss(pred_bbox[b].unsqueeze(0),
                                   V(gt_bbox))

            if old_loss is None: old_loss = (prob_loss, bbox_loss)

            if ((prob_loss + bbox_loss) < (old_loss[0] + old_loss[1])).data.cpu().numpy()[0]:
                old_loss = (prob_loss, bbox_loss)

        p_loss.append(old_loss[0])
        b_loss.append(old_loss[1])

    prob_loss = 0
    bbox_loss = 0
    for p in p_loss:
        prob_loss += p
    prob_loss /= len(p_loss)
    for p in b_loss:
        bbox_loss += p
    bbox_loss /= len(b_loss)

    return prob_loss, bbox_loss
