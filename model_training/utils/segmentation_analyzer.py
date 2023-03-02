import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
from ummon.logger import Logger
import ummon.utils as uu
from ummon.analyzer import Analyzer
import time
from ummon.predictor import Predictor
from utils.metrics import binarize, inters_over_union, pixel_accuracy, specificity, precision, f1, sensitivity 
import numpy as np


class SegmentationAnalyzer(Analyzer):
    """
    This class is a subclass of the generic analyzer class.  It add additional 
    metrics to the metric dictionary.
    
    Methods
    -------
    evaluate()          : Evaluates a model with given validation dataset
    """
            
    @staticmethod    
    def evaluate(model, loss_function, dataset, batch_size=-1, compute_accuracy=True, limit_batches=-1, logger=Logger(), metrics=[], is_validation=False):
        """
        Evaluates a model with given validation dataset
        
        Arguments
        ---------
        model           : nn.module
                          The model
        loss_function   : nn.module
                          The loss function to evaluate
        dataset         : torch.utils.data.Dataset OR tuple (X,y, (bs))
                          Dataset to evaluate
        batch_size      : int
                          batch size used for evaluation (default: -1 == ALL)
        compute_accuracy : bool
                          specifies if the output gets classified
        limit_batches   : int
                          specified if only a limited number of batches shall be analyzed. Useful for testing a subset of the training data.
        logger          : ummon.Logger (Optional)
                          The logger to be used for output messages
        
        Return
        ------
        Dictionary
        A dictionary containing keys `loss`, `accuracy`, ´samples_per_second`, `detailed_loss`, 'args[]`
        """

        dataloader = uu.gen_dataloader(dataset, batch_size=batch_size, has_labels=True, logger=logger)
        t = time.time()
        loss_average, acc_average, iou_average, sensitivity_average, specificity_average, precision_average, f1_average = 0., 0., 0., 0., 0., 0., 0.
        use_cuda = next(model.parameters()).is_cuda
        device = "cuda" if use_cuda else "cpu"
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):

             # limit
                if limit_batches == i:
                    break

                # data
                inputs, labels = uu.input_of(batch), uu.label_of(batch)
               
                # Handle cuda
                inputs, labels = uu.tuple_to(inputs, device), uu.tuple_to(labels, device)
                output = model(inputs)
                
                # Unsqueeze targets if necessary (may occur with class=1 segmentations)
                if output.dim() > labels.dim():
                    labels.unsqueeze_(dim=1)

                # Compute output
                try:
                    loss = loss_function(output, labels)
                except(ValueError):
                    # case: targets are not formatted correctly
                    loss = loss_function(output, labels.view_as(output))


                loss_average = uu.online_average(loss, i + 1, loss_average)

                if compute_accuracy == True:
                    classes = binarize(output.cpu(), activation="sigmoid")
                    acc = pixel_accuracy(classes, labels.cpu())
                    acc_average = uu.online_average(acc, i+1, acc_average)
                    iou = inters_over_union(classes, labels.cpu())
                    iou_average = uu.online_average(iou, i+1, iou_average)
                    sensitivity_score = sensitivity(classes, labels.cpu())
                    sensitivity_average = uu.online_average(sensitivity_score, i+1, sensitivity_average)
                    specificity_score = specificity(classes,labels.cpu())
                    specificity_average = uu.online_average(specificity_score, i+1, specificity_average)
                    precision_score = precision(classes, labels.cpu())
                    precision_average = uu.online_average(precision_score, i+1, precision_average)
                    f1_score = f1(classes, labels.cpu())
                    f1_average = uu.online_average(f1_score, i+1, f1_average)


        
            # save results in dict
            evaluation_dict = {}
            evaluation_dict["accuracy"] = acc_average
            evaluation_dict["samples_per_second"] = len(dataloader) / (time.time() - t)
            evaluation_dict["loss"] = loss_average
            evaluation_dict["detailed_loss"] = {"__repr__(loss)" : repr(loss_function)}
            evaluation_dict["iou"] = iou_average
            evaluation_dict["sensitivity"] = sensitivity_average
            evaluation_dict["specificity"] = specificity_average
            evaluation_dict["precision"] = precision_average
            evaluation_dict["f1"] = f1_average
        
        return evaluation_dict
    
    # output evaluation string for regression
    @staticmethod
    def evalstr(learningstate):
        
        # without validation data
        if learningstate.has_validation_data():
            return 'loss (trn): {:4.5f}, lr={:1.5f}'.format(
                learningstate.current_training_loss(), 
                learningstate.current_lrate())
        
        # with validation data
        else:
            return 'loss(trn/val):{:4.5f}/{:4.5f}, lr={:1.5f}{}'.format(
                learningstate.current_training_loss(), 
                learningstate.current_validation_loss(),
                learningstate.current_lrate(),
                ' [BEST]' if learningstate.is_best_validation_model() else '')


    @staticmethod
    def evalstr(learningstate):
        # without validation data
        if learningstate.has_validation_data():
            return 'loss (trn): {:4.5f}, lr={:1.5f}'.format(
                learningstate.current_training_loss(), 
                learningstate.current_lrate())
        
        # with validation data
        else:
            return 'loss(trn/val):{:4.5f}/{:4.5f}, acc(val):{:.2f}%, lr={:1.5f}{}'.format(
                learningstate.current_training_loss(), 
                learningstate.current_validation_loss(),
                learningstate.current_validation_acc()*100,
                learningstate.current_lrate(),
                ' [BEST]' if learningstate.is_best_validation_model() else '')
