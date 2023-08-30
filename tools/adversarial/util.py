
import numpy as np

def get_gt_bboxes_scores_and_labels(Anns, cat2label, img_name, scale_factor, ncls, scale_flag=None):
    bboxes = []
    scores = []
    labels = []

    if '/' in img_name:
        img_name = img_name.split('/')[-1]
    # modified
    for i in Anns.keys():
        if type(i) == int :
            img_id = int(img_name.split('.')[0])
        else:
            img_id = img_name.split('.')[0]
        break

    img2bboxes = [Anns[img_id][i]['bbox'] for i in range(len(Anns[img_id]))]
    img2labels = [cat2label[Anns[img_id][i]['category_id']]  for i in range(len(Anns[img_id]))]
    if scale_flag:
        img2bboxes = np.array(img2bboxes*scale_factor)
    img2bboxes = np.array(img2bboxes)
    xs_left, ys_left, ws, hs = img2bboxes[:, 0], img2bboxes[:, 1], img2bboxes[:, 2], img2bboxes[:, 3]
    bboxes = np.column_stack((xs_left, ys_left, xs_left+ws, ys_left+hs))
    labels = np.array(img2labels)
    scores = np.zeros((len(labels), ncls))
    for i in range(len(labels)):
        scores[i, labels[i]] = 1.0
    return bboxes, scores, labels
