# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np

class ConfusionMatrix(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        # axis = 0: prediction
        # axis = 1: target
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        hist = np.zeros((n_class, n_class))
        hist[label_pred, label_true] += 1

        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            tmp = self._fast_hist(lt.item(), lp.item(), self.n_classes)
            self.confusion_matrix += tmp

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - sensitivity
            - specificity
            - fwavacc
        """
        hist = self.confusion_matrix
        # accuracy is recall/sensitivity for each class, predicted TP / all real positives
        # axis in sum: perform summation along

        if sum(hist.sum(axis=1)) != 0:
            acc = sum(np.diag(hist)) / sum(hist.sum(axis=1))
            sn = hist[0, 0] / hist.sum(axis=1)[0] #TP/(TP+FN)
            sp = hist[self.n_classes-1, self.n_classes-1] / hist.sum(axis=1)[self.n_classes-1] #TN/(TN+FP)
            pr = hist[0, 0] / hist.sum(axis=0)[0] #TP/(TP+FP)
            f1 = 2 * (pr*sn) / (pr+sn) # 2 * (precision * recall) / (precision + recall)
        else:
            acc = 0.0
            sn = 0.0
            sp = 0.0
            pr = 0.0
            f1 = 0.0
            
        
        return {'accuracy': acc,
                'sensitivity': sn,
                'specificity': sp,
                'precision': pr,
                'fmeasure': f1,
                }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
