import time
import numpy as np
import types
import torch.nn as nn
import torch.utils.data
import ummon.utils as uu
from ummon.trainer import Trainer
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.dataset import Subset
from ummon.logger import Logger
from ummon.schedulers import *
from ummon.trainingstate import *
from ummon.analyzer import Analyzer

__all__ = ["SegmentationTrainer", " SegmentationLogger"]

class SegmentationTrainer(Trainer):
    """
    This class is a subclass of the generic trainer class.
    It contains the possiblity to use an additional logger to keep track of different
    segmentation metrics over all epochs.
    
    Constructor
    -----------
    logger            : ummon.Logger
                        The logger to use (if NULL default logger will be used)
    model             : torch.nn.module
                        The model used for training
    loss_function     : torch.nn.module
                      : The loss function for the optimization
    optimizer         : torch.optim.optimizer
                        The optimizer for the training
    trainingstate     : ummon.Trainingstate
                        A previously instantiated training state variable. Can be used 
                        to initialize model and optimizer with a previous saved state.
    scheduler         : torch.optim.lr_scheduler._LRScheduler
                        OPTIONAL A learning rate scheduler
    model_filename    : String
                      : OPTIONAL Name of the persisted model files
    model_keep_epochs : bool
                        OPTIONAL Specifies intermediate (for every epoch) model persistency (default False).
    precision         : np.dtype
                        OPTIONAL Specifiec FP32 or FP64 Training (default np.float32).
    convergence_eps   : float
                        OPTIONAL Specifies when the training has converged (default np.float32.min).
    combined_training_epochs : int
                        OPTIONAL Specifies how many epochs combined retraining (training and validation data) shall take place 
                            after the usal training cycle (default 0). 
    additional_logger: OPTIONAL Additional logger to keep track of epochs which were discarded by early stopping scheduler                    
    
    Methods
    -------
    fit()            :  trains a model
    _evaluate()      :  validates the model
    _moving_average():  helper method
             
    """
    def __init__(self, *args, **kwargs):
        
        # required arguments
        if len(args) != 4:
            raise ValueError('You must provide at least a logger, model, loss_function and optimizer for training.') 
        self.logger = args[0]
        assert type(self.logger) == Logger
        self.model = args[1]
        assert isinstance(self.model, nn.Module)
        self.criterion = args[2]
        assert isinstance(self.criterion, nn.Module) or isinstance(self.criterion, types.LambdaType)
        self.optimizer = args[3]
        assert isinstance(self.optimizer, torch.optim.Optimizer)
        
        # defaults
        self.trainingstate = Trainingstate(filename=None, model_keep_epochs=False)
        self.scheduler = None
        self.precision = np.float32
        self.convergence_eps = np.finfo(np.float32).min
        self.combined_training_epochs = False
        
        # optional arguments
        for key in kwargs:
            if key == 'trainingstate':
                if not isinstance(kwargs[key], Trainingstate):
                    raise TypeError('{} is not a training state'.format(type(kwargs[key]).__name__))
                self.trainingstate = kwargs[key]
                if 'model_filename' in kwargs.keys():
                    self.trainingstate.filename = str(kwargs["model_filename"]).split(self.trainingstate.extension)[0]            
                if 'model_keep_epochs' in kwargs.keys():     
                    self.trainingstate.model_keep_epochs = int(kwargs["model_keep_epochs"])
            elif key == 'scheduler':
                if not isinstance(kwargs[key], torch.optim.lr_scheduler._LRScheduler) and not \
                    isinstance(kwargs[key], StepLR_earlystop):
                    raise TypeError('{} is not a scheduler'.format(type(kwargs[key]).__name__))
                if isinstance(kwargs[key], StepLR_earlystop) and 'trainingstate' not in kwargs.keys():
                    raise ValueError('StepLR_earlystop needs an external Trainingstate (you provided None).')
                self.scheduler = kwargs[key]
            elif key == 'precision':
                assert kwargs[key] == np.float32 or kwargs[key] == np.float64
                self.precision = kwargs[key]
            elif key == 'convergence_eps':
                self.convergence_eps = float(kwargs[key])
            elif key == 'combined_training_epochs':
                if self.trainingstate.filename is None:
                    raise ValueError('Combined retraining needs a model_filename to load the best model after training. (you provided None).')
                self.combined_training_epochs = int(kwargs[key])
            elif key == 'model_keep_epochs':
                self.trainingstate.model_keep_epochs = int(kwargs[key])
            elif key == 'model_filename':
                self.trainingstate.filename = str(kwargs[key]).split(self.trainingstate.extension)[0]            
            elif key == 'use_cuda':
                self.cuda = kwargs[key]
            elif key == 'additional_logger':
                self.additional_logger = kwargs[key]
            else:
                raise ValueError('Unknown keyword {} in constructor.'.format(key))
        
        # Training parameters
        self.epoch = 0
        
        # training state was filled by previous training or by persisted training state
        if 'trainingstate' in kwargs.keys() and self.trainingstate.state != None:
            self._status_summary()
            self.epoch = self.trainingstate.state["training_loss[]"][-1][0]
            self.trainingstate.load_optimizer_(self.optimizer)
            self.trainingstate.load_weights_(self.model, self.optimizer)
            if isinstance(self.scheduler, StepLR_earlystop):
                self.trainingstate.load_scheduler_(self.scheduler)
        
        # Computational configuration
        self.use_cuda = next(self.model.parameters()).is_cuda
        if self.use_cuda:
            if not torch.cuda.is_available():
                self.logger.error('CUDA is not available on your system.')
        self.model = Trainingstate.transform_model(self.model, self.optimizer, 
            self.precision, self.use_cuda)
    
    
    def _evaluate_training(self, Analyzer, batch, batches, 
                  time_dict, 
                  epoch,  
                  dataloader_validation, 
                  dataloader_training,
                  eval_batch_size, metrics):
        """
        Evaluates the current training state against agiven analyzer
        
        Arguments
        ---------
        *Analyzer (ummon.Analyzer) : A training type specific analyzer.
        *batch (int) : The batch number.
        *batches (int) : The total number of batches.
        *time_dict (dict) : Dictionary that is used for profiling executing time.
        *epoch (int) : The current epoch.
        *eval_interval (int) : The interval in epochs for evaluation against validation dataset.
        *dataloader_validation (torch.utils.data.Dataloader) : Validation data.
        *dataloader_training : The training data
        *eval_batch_size (int) : A custom batch size to be used during evaluation
        
        """
        
        # EVALUATE ON TRAINING SET
        validation_loss = None
        evaluation_dict_train = Analyzer.evaluate(self.model, 
                                                self.criterion, 
                                                dataloader_training,
                                                batch_size=eval_batch_size,
                                                limit_batches=200,
                                                logger=self.logger,
                                                metrics=metrics)

        self.additional_logger.update_trn({"epoch" : epoch +1, "batch_size" : eval_batch_size, "lr":self.optimizer.param_groups[0]['lr']})  
        self.additional_logger.update_trn(evaluation_dict_train)
        
        # Log epoch
        if dataloader_validation is not None:
            
                # MODEL EVALUATION
                evaluation_dict = Analyzer.evaluate(self.model, 
                                                    self.criterion, 
                                                    dataloader_validation, 
                                                    batch_size=eval_batch_size,
                                                    limit_batches=-1,
                                                    logger=self.logger,
                                                    metrics=metrics)
        
                # UPDATE TRAININGSTATE
                self.trainingstate.update_state(epoch + 1, self.model, self.criterion, self.optimizer, 
                                        training_loss = evaluation_dict_train["loss"], 
                                        training_accuracy = evaluation_dict_train["accuracy"],
                                        training_dataset = dataloader_training.dataset,
                                        training_batchsize = dataloader_training.batch_size,
                                        trainer_instance = type(self),
                                        precision = self.precision,
                                        detailed_loss = evaluation_dict["detailed_loss"],
                                        validation_loss = evaluation_dict["loss"],
                                        validation_accuracy = evaluation_dict["accuracy"],  
                                        validation_dataset = dataloader_validation.dataset,
                                        samples_per_second = evaluation_dict["samples_per_second"],
                                        scheduler = self.scheduler,
                                        combined_retraining = self.combined_training_epochs,
                                        evaluation_dict_train=evaluation_dict_train,
                                        evaluation_dict_eval=evaluation_dict)

                self.additional_logger.update_val({"epoch" : epoch +1})  
                self.additional_logger.update_val(evaluation_dict) 
                                   
                validation_loss = evaluation_dict["loss"]
                                
        else: # no validation set
            
            evaluation_dict = evaluation_dict_train

            self.trainingstate.update_state(epoch + 1, self.model, self.criterion, self.optimizer, 
                training_loss = evaluation_dict_train["loss"], 
                training_accuracy = evaluation_dict_train["accuracy"],
                training_batchsize = dataloader_training.batch_size,
                training_dataset = dataloader_training.dataset,
                trainer_instance = type(self),
                precision = self.precision,
                detailed_loss = repr(self.criterion),
                scheduler = self.scheduler,
                evaluation_dict_train=evaluation_dict_train)

        self.trainingstate.evalstr()
        self.logger.log_epoch(epoch + 1, batch + 1,
                                batches, 
                                dataloader_training.batch_size,
                                time_dict,
                                self.trainingstate)

        return validation_loss
    
    
class SegmentationLogger:
    def __init__(self, keys):
        self.record_trn = {key: [] for key in keys}
        self.record_val = {key: [] for key in keys}
        
    def update_trn(self, dictionary):
        if isinstance(dictionary, dict):
            common_keys = self.record_trn.keys() & dictionary.keys()
            filtered_dict = { key: dictionary[key] for key in common_keys }
            for key,value in filtered_dict.items():
                self.record_trn[key] = [*self.record_trn[key],value]
        else:    
            raise NotImplementedError
            
    def update_val(self, dictionary):
        if isinstance(dictionary, dict):
            common_keys = self.record_val.keys() & dictionary.keys()
            filtered_dict = { key: dictionary[key] for key in common_keys }
            for key,value in filtered_dict.items():
                self.record_val[key] = [*self.record_val[key],value]
        else:    
            raise NotImplementedError

    def combine(self):
        temp_trn = {k+'_trn': v for k, v in self.record_trn.items()}
        temp_val = {k+'_val': v for k, v in self.record_val.items()}
        temp_combined = {**temp_val, **temp_trn}
        record_combined = {k: v for k, v in temp_combined.items() if k !="epoch_val" and k !="lr_val"}
        return record_combined

    def save_results(self, path):
        import json
        data = self.combine()
        # Serialize data into file:
        json.dump(data, open(path, 'w' ))
        
