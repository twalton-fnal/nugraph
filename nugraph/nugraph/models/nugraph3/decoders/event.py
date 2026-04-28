"""NuGraph3 event decoder"""
from typing import Any
import matplotlib.pyplot as plt
import seaborn as sn
import tempfile
import torch
from torch import nn
import torchmetrics as tm
from torch_geometric.data import Batch
from pytorch_lightning.loggers import Logger, TensorBoardLogger

from ....util import ConfusionMatrixLogger, RecallLoss
from ....util.DANNLoss import ReverseLayerF
from ....util.EmbeddingPlotter import CombinedEmbeddingPlot
from ..types import Data

class EventDecoder(nn.Module):
    """
    NuGraph3 event decoder module, which includes the option
    to enable the Domain Adaptation on event-level features

    Convolve the interaction node embedding down to a set of categorical scores
    for each event class.

    Args:
        interaction_features: Number of interaction node features
        event_classes: List of event classes
    """
    def __init__(self,
                 interaction_features: int,
                 event_classes: list[str],
                 da_loss_fnc_name: str = None,
                 warmup_epochs: int = 0):
        super().__init__()

        self.warmup_epochs = warmup_epochs
        self.da_loss_fnc_name = da_loss_fnc_name
        self.use_domain_adaptation = False 
        self.domain_adaptation_classes = [ "dann", "mmd", "semantic", "sinkhorn" ]
        
        # loss function
        self.loss = RecallLoss()
        if self.da_loss_fnc_name == "dann":
           self.loss_dann = nn.CrossEntropyLoss()  
           loss_dann = torch.tensor(0.0)    
        
        # temperature parameter
        self.source_temp = nn.Parameter(torch.tensor(0.))
        if self.da_loss_fnc_name in self.domain_adaptation_classes: 
           self.target_temp = nn.Parameter(torch.tensor(0.))
           self.da_temp = nn.Parameter(torch.tensor(0.))

        # metrics
        metric_args = {
            "task": "multiclass",
            "num_classes": len(event_classes)
        }
        self.source_recall = tm.Recall(**metric_args)
        self.source_precision = tm.Precision(**metric_args)
        self.source_cm_recall = tm.ConfusionMatrix(normalize="true", **metric_args)
        self.source_cm_precision = tm.ConfusionMatrix(normalize="pred", **metric_args)
        self.cm_logger = ConfusionMatrixLogger(event_classes)
        
        if self.da_loss_fnc_name in self.domain_adaptation_classes:
           self.target_recall = tm.Recall(**metric_args)
           self.target_precision = tm.Precision(**metric_args)
           self.target_cm_recall = tm.ConfusionMatrix(normalize="true", **metric_args)
           self.target_cm_precision = tm.ConfusionMatrix(normalize="pred", **metric_args)
           self.embeddings = CombinedEmbeddingPlot(method="umap") 
        
        # network
        self.net = nn.Linear(in_features=interaction_features,
                             out_features=len(event_classes))

        self.classes = event_classes

        # Domain classifier network for DANN
        if self.da_loss_fnc_name == "dann":
           self.domain_net = nn.Linear(in_features=interaction_features,
                                       out_features=2)
           self.domain_classes = ['source', 'target']

    
    def forward(self, data: list[Data], stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 event decoder forward pass

        Args:
            data: Graph data object or both the source and target data objects
            stage: Stage name (train/val/test)
        """
        
        # the data
        if self.da_loss_fnc_name == None:
           source_data = data
        else:
           source_data, target_data = data 

        # run network and calculate loss
        loss = loss_source = loss_target = lossDA = raw_lossDA = 0
        
        x_source = self.net(source_data["evt"].x)
        y_source = source_data["evt"].y
        w_source = 2 * (-1 * self.source_temp).exp()
        loss_no_w_source = self.loss(x_source, y_source)
        loss_source = w_source * loss_no_w_source + self.source_temp
        
        if self.da_loss_fnc_name == None:
           loss = loss_source
            
        if self.da_loss_fnc_name in self.domain_adaptation_classes:
           x_target = self.net(target_data["evt"].x)
           y_target = target_data["evt"].y
           w_target = 2 * (-1 * self.target_temp).exp()  # SHOULD TARGET HAVE ITS OWN WEIGHT?
           loss_target = w_target * self.loss(x_target, y_target) + self.target_temp           

        """
           Domain Adaptation is implemented via the Domain-Adversarial Neural Network (DANN). 
           Both source and target labels are being used to calculate event losses (loss_source and loss_target).
           All losses are scaled by their own temperatures, which are trainable parameters.
           DA loss is also capped at most 1/4 of the loss_source (to be on the safe side). 
        """ 
        if self.use_domain_adaptation and self.da_loss_fnc_name == "dann":  
           # Domain Classification
           alpha = 1   # Weight of DA is handeled by weight wDA so alpha=1 inside of the gradient reversal layer
           reversed_S = ReverseLayerF.apply(source_data["evt"].x, alpha)
           xSS = self.domain_net(reversed_S)
           ySS = torch.zeros(xSS.shape[0]).type(torch.LongTensor)
    
           reversed_T = ReverseLayerF.apply(target_data["evt"].x, alpha)
           xTT = self.domain_net(reversed_T)
           yTT = torch.ones(xTT.shape[0]).type(torch.LongTensor)
    
           combined_image = torch.cat((xSS, xTT), 0).cuda()
           combined_label = torch.cat((ySS, yTT), 0).cuda()

           wDA = 2 * (-1 * self.da_temp).exp()
           raw_lossDA = wDA * self.loss_dann(combined_image, combined_label) + self.da_temp 

           # Smooth capping of the DA based on the source event loss value (currently to be at most 1/4 of the event loss value)
           sharp = 20.0  # sharpness of transition when source event loss goes from positive to negative
           sig = torch.sigmoid(sharp * loss_source)
           max_lossDA = sig * (loss_source / 4) + (1 - sig) * (4 * loss_source)
           lossDA = torch.min(raw_lossDA, max_lossDA)

           # Total loss
           loss = loss_source + loss_target + lossDA

        # calculate metrics
        metrics = {}
        if stage:
            if self.da_loss_fnc_name == None:
               metrics[f"event/loss-{stage}"] = loss
               metrics[f"event/recall-{stage}"] = self.source_recall(x_source, y_source)
               metrics[f"event/precision-{stage}"] = self.source_precision(x_source, y_source)
            else:
               metrics[f"loss_event_total/{stage}"] = loss
               metrics[f"recall_event_source/{stage}"] = self.source_recall(x_source, y_source)
               metrics[f"precision_event_source/{stage}"] = self.source_precision(x_source, y_source)
               metrics[f"recall_event_target/{stage}"] = self.target_recall(x_target, y_target)
               metrics[f"precision_event_target/{stage}"] = self.target_precision(x_target, y_target)
               metrics[f"loss_event_source/{stage}"] = loss_source
               metrics[f"loss_event_target/{stage}"] = loss_target
               metrics[f"Using_DA_or_not/{stage}"] = 1

               if self.da_loss_fnc_name == "dann":
                  metrics[f"DA_loss_capped_event/{stage}"] = lossDA
                  metrics[f"DA_loss_uncapped_event/{stage}"] = raw_lossDA 
            
        if stage == "train":
            if self.da_loss_fnc_name == None:
               metrics["temperature/event"] = self.source_temp
            else:
               metrics["temperature/event_source"] = self.source_temp
               metrics["temperature/event_target"] = self.target_temp
               metrics["temperature/event_DA"] = self.da_temp
            
        if stage in ["val", "test"]:
            self.source_cm_recall.update(x_source, y_source)
            self.source_cm_precision.update(x_source, y_source)
            
            if self.da_loss_fnc_name in self.domain_adaptation_classes:
               self.target_cm_recall.update(x_target, y_target)
               self.target_cm_precision.update(x_target, y_target)
               self.embeddings.update(source_data["evt"].x, y_source, target_data["evt"].x, y_target)

        # add inference output to graph object
        source_data["evt"].e = x_source.softmax(dim=1)
        if isinstance(source_data, Batch):
            # pylint: disable=protected-access
            source_data._slice_dict["evt"]["e"] = source_data["evt"].ptr
            inc = torch.zeros(source_data.num_graphs, device=source_data["evt"].x.device)
            source_data._inc_dict["evt"]["e"] = inc

        if self.da_loss_fnc_name in self.domain_adaptation_classes:
           target_data["evt"].e = x_target.softmax(dim=1)
           if isinstance(target_data, Batch):
            target_data._slice_dict["evt"]["e"] = target_data["evt"].ptr
            incT = torch.zeros(source_data.num_graphs, device=target_data["evt"].x.device)
            target_data._inc_dict["evt"]["e"] = incT
        
        return loss, metrics

    def draw_confusion_matrix(self, cm: tm.ConfusionMatrix) -> plt.Figure:
        """
        Draw a confusion matrix

        Args:
            cm: Confusion matrix object
        """
        confusion = cm.compute().cpu()
        fig = plt.figure(figsize=[8,6])
        sn.heatmap(confusion,
                   xticklabels=self.classes,
                   yticklabels=self.classes,
                   vmin=0, vmax=1,
                   annot=True)
        plt.ylim(0, len(self.classes))
        plt.xlabel("Assigned label")
        plt.ylabel("True label")
        return fig

    
    def on_epoch_end(self, logger: Logger | list[Logger], stage: str,
                     epoch: int) -> None: # pylint: disable=unused-argument
        """
        NuGraph3 decoder end-of-epoch callback function

        Args:
            logger: PyTorch Lightning logger object(s)
            stage: Training stage
            epoch: Training epoch index
        """

        if self.da_loss_fnc_name == None:        
           self.cm_logger.log(f"event/recall-matrix-{stage}",
                              self.source_cm_recall, logger, epoch)
           self.cm_logger.log(f"event/precision-matrix-{stage}",
                              self.source_cm_precision, logger, epoch)
            
        else:
           logger.experiment.add_figure(f"recall_event_matrix_source/{stage}",
                                        self.draw_confusion_matrix(self.source_cm_recall),
                                        global_step=epoch)
           self.source_cm_recall.reset()

           logger.experiment.add_figure(f"recall_event_matrix_target/{stage}",
                                        self.draw_confusion_matrix(self.target_cm_recall),
                                        global_step=epoch)
           self.target_cm_recall.reset()

           logger.experiment.add_figure(f"precision_event_matrix_source/{stage}",
                                        self.draw_confusion_matrix(self.source_cm_precision),
                                        global_step=epoch)
           self.source_cm_precision.reset()

           logger.experiment.add_figure(f"precision_event_matrix_target/{stage}",
                                        self.draw_confusion_matrix(self.target_cm_precision),
                                        global_step=epoch)
           self.target_cm_precision.reset()
        
           # Plot the embedding space 
           dat1, lab1, dat2, lab2 = self.embeddings.compute()
           dat1sub, lab1sub = self.embeddings.subsample(dat1, lab1, max_samples=1000)
           dat2sub, lab2sub = self.embeddings.subsample(dat2, lab2, max_samples=1000)
        
           embeddings_fig = self.embeddings.plot_combined(dat1sub, lab1sub, dat2sub, lab2sub, 
                                                          epoch=epoch, class_names=self.classes)
           logger.experiment.add_figure(f"Embeddings event/{stage}",
                                        embeddings_fig, global_step=epoch)
           self.embeddings.reset()
