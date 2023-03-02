import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def inters_over_union(pred, target):
    '''intersection over union = TP / (TP + FP + FN)
    if there are no TP, no FP and no FN the score is assumed to be 1'''
    p_bool = np.array(pred) == 1
    t_bool = np.array(target) == 1
    intersection = np.sum(np.logical_and(p_bool, t_bool))
    union = np.sum(np.logical_or(p_bool, t_bool))
    if union == 0:
        iou_score=1.0
    else:
        iou_score = intersection / union
    return iou_score

def pixel_accuracy(pred, target):
    '''accuracy =  (TP + TN) / (TP + TN + FP + FN)'''
    p_bool = np.array(pred) == 1
    t_bool = np.array(target) == 1
    correct = p_bool == t_bool
    accuracy_score = np.sum(correct) / t_bool.size
    return accuracy_score

def sensitivity(pred, target):
    '''sensitivity/recall = TP / (TP + FN) 
    if there are no TPs and FNs the presence of FPs determines the score
    '''
    p_bool = np.array(pred) == 1
    t_bool = np.array(target) == 1 
    true_pos = np.sum(np.logical_and(p_bool, t_bool) )  
    false_neg = np.sum(np.logical_and((~p_bool), t_bool))
    if (true_pos + false_neg) == 0: 
        false_pos = np.sum(np.logical_and(p_bool, ~t_bool))
        sensitivity_score = 1.0 if false_pos == 0 else 0.0
    else:
        sensitivity_score = true_pos / (true_pos + false_neg)
    return sensitivity_score

def specificity(pred, target):
    '''specificity = TN / (TN + FP) 
    if there are no TNs and FPs the presence of FNs determines the score
    '''
    p_bool = np.array(pred) == 1
    t_bool = np.array(target) == 1 
    true_neg = np.sum(np.logical_and(~p_bool, ~t_bool))   
    false_pos = np.sum(np.logical_and(p_bool, ~t_bool))
    if (true_neg + false_pos) == 0: 
        false_neg = np.sum(np.logical_and(~p_bool, t_bool))
        specificity_score = 1.0 if false_neg == 0 else 0.0
    else:
        specificity_score = true_neg / (true_neg + false_pos)
    return specificity_score

def precision(pred, target):
    '''precision = TP / (TP + FP) 
    if there are no TPs and FPs the presence of FNs determines the score
    '''
    p_bool = np.array(pred) == 1
    t_bool = np.array(target) == 1 
    true_pos = np.sum(np.logical_and(p_bool, t_bool)) 
    false_pos = np.sum(np.logical_and(p_bool, ~t_bool))
    if (true_pos + false_pos) == 0: 
        false_neg = np.sum(np.logical_and(~p_bool, t_bool))
        precision_score = 1.0 if false_neg == 0 else 0.0
    else:
        precision_score = true_pos / (false_pos + true_pos)
    return precision_score

def f1(pred, target):
    '''f1= 2*(precision * sensitivity) / (precision + sensitivity)
    if precision and sensitivity are 0 the score is assumed to be 0'''

    precision_score = precision(pred,target)
    sensitivity_score = sensitivity(pred, target)
    if (precision_score + sensitivity_score) == 0: 
        f1_score = 0.0
    else:
        f1_score = 2* (precision_score * sensitivity_score) / (precision_score + sensitivity_score)
    return f1_score

def binarize(output, activation=None, threshold=0.5):
    ''' Converts torch tensor to a 0 and 1 only representation 
    if desired the input array is run through a none linarity before the thresholding.
    Available are "sigmoid" or "softmax" '''
    # If desired add a elementwise none linearity to the output
    if activation=="sigmoid":
            output = torch.sigmoid(Variable(output)).data        
    elif activation=="softmax":
            output = F.softmax(Variable(output), dim=1).data
    elif activation is not None:
        raise NotImplementedError

    output[output>=threshold] = 1
    output[output<threshold] = 0
    return output