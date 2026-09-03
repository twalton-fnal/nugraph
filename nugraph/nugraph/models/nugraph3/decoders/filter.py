"""NuGraph3 filter decoder"""
from typing import Any
import tempfile
import matplotlib.pyplot as plt
import seaborn as sn

import torch
from torch import nn
import torchmetrics as tm
from torch_geometric.data import Batch
from pytorch_lightning.loggers import Logger

from ....util import ConfusionMatrixLogger
from ....util.DANNLoss import ReverseLayerF
from ....util.MMDLoss import MMDLoss
from ....util.SemanticLoss import SemanticAlignmentLoss
from ....util.SinkhornLoss import Sinkhorn
from ....util.EmbeddingPlotter import CombinedEmbeddingPlot

from ..types import Data

class FilterDecoder(nn.Module):
    """
    NuGraph3 filter decoder module

    Convolve hit node embedding down to a single node score to identify and
    filter out graph nodes that are not part of the primary physics
    interaction.

    Args:
        hit_features: Number of hit node features
    """
    def __init__(self,
                 hit_features: int,
                 da_loss_fnc_name: str = None,
                 warmup_epochs: int = 0):
        super().__init__()

        self.warmup_epochs = warmup_epochs
        self.use_da_after_warmups = False
        self.da_loss_fnc_name = da_loss_fnc_name
        self.use_domain_adaptation = False 
        self.domain_adaptation_classes = [ "dann", "mmd", "semantic", "sinkhorn" ]
   
        # loss function
        self.loss = nn.BCEWithLogitsLoss()
        if self.da_loss_fnc_name == "dann":
           self.loss_dann = nn.CrossEntropyLoss()  
           loss_dann = torch.tensor(0.0)    

        # temperature parameter
        self.source_temp = nn.Parameter(torch.tensor(0.))
        if self.da_loss_fnc_name in self.domain_adaptation_classes: 
           self.target_temp = nn.Parameter(torch.tensor(0.))
           self.da_temp = nn.Parameter(torch.tensor(0.))
       
        # metrics
        self.classes = [ "noise", "signal" ] 
        metric_args  = {"task": "binary"}

        self.source_recall = tm.Recall(**metric_args)
        self.source_precision = tm.Precision(**metric_args)
        self.source_cm_recall = tm.ConfusionMatrix(normalize="true", **metric_args)
        self.source_cm_precision = tm.ConfusionMatrix(normalize="pred", **metric_args)
        self.cm_logger = ConfusionMatrixLogger(self.classes)

        if self.da_loss_fnc_name in self.domain_adaptation_classes:
           self.target_recall = tm.Recall(**metric_args)
           self.target_precision = tm.Precision(**metric_args)
           self.target_cm_recall = tm.ConfusionMatrix(normalize="true", **metric_args)
           self.target_cm_precision = tm.ConfusionMatrix(normalize="pred", **metric_args)

        # network
        self.net = nn.Linear(hit_features, 1)

        # Domain classifier network for DANN
        if self.da_loss_fnc_name == "dann":  
           self.domain_net = nn.Sequential(
                                  nn.Linear(in_features=hit_features,out_features=64),
                                  nn.ReLU(),
                                  nn.Linear(in_features=64,out_features=2)
           )  
           self.domain_classes = ['source', 'target']

        

    def forward(self, data: Data, stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 filter decoder forward pass

        Args:
            data: Graph data object
            stage: Stage name (train/val/test)
        """

        # the data
        if not self.da_loss_fnc_name in self.domain_adaptation_classes:
           source_data = data
        else:
           source_data, target_data = data 
            
        # run network and add output to graph object      
        def _run_net_and_add_output(data: Any, net: torch.nn.Module, key: str):
            data[key].x_filter = net(data[key].x)
            if isinstance(data, Batch):
               data._slice_dict[key]["x_filter"] = data[key].ptr
               inc = torch.zeros(data.num_graphs, device=data[key].x.device)
               data._inc_dict[key]["x_filter"] = inc

        _run_net_and_add_output(source_data, self.net, "hit")
        if self.da_loss_fnc_name in self.domain_adaptation_classes:
           _run_net_and_add_output(target_data, self.net, "hit")

        # calculate loss
        loss = loss_source = loss_target = lossDA = loss_func = raw_lossDA = 0

        #x = self.net(data["hit"].x).squeeze(dim=-1)
        x_source = source_data["hit"].x_filter.squeeze(dim=-1)
        y_source = (source_data["hit"].y_semantic != -1).float()
        w_source = 2 * (-1 * self.source_temp).exp()
        loss_source = w_source * self.loss(x_source, y_source) + self.source_temp

        if self.da_loss_fnc_name in self.domain_adaptation_classes:
           x_target = target_data["hit"].x_filter.squeeze(dim=-1)
           y_target = (target_data["hit"].y_semantic != -1).float()
           w_target = 2 * (-1 * self.target_temp).exp()
           loss_target = w_target * self.loss(x_target, y_target) + self.target_temp
            
        if self.da_loss_fnc_name == None:
           loss = loss_source

        if self.use_domain_adaptation:
           wDA = 2 * (-1 * self.da_temp).exp()
            
           if self.da_loss_fnc_name == "dann":    
              alpha = 1
               
              reversed_S = ReverseLayerF.apply(source_data["hit"].x, alpha)
              xSS = self.domain_net(reversed_S)
              ySS = torch.zeros(xSS.shape[0], dtype=torch.long, device=xSS.device)
            
              reversed_T = ReverseLayerF.apply(target_data["hit"].x, alpha)
              xTT = self.domain_net(reversed_T) 
              yTT = torch.ones(xTT.shape[0], dtype=torch.long, device=xTT.device)    

              combined_image = torch.cat((xSS, xTT), dim=0)  
              combined_label = torch.cat((ySS, yTT), dim=0)

              loss_func = self.loss_dann(combined_image, combined_label)
              raw_lossDA = wDA * loss_func + self.da_temp

           else:
              sys.exit( f"The function {self.da_loss_fnc_name} does not exist." )

           lossDA = raw_lossDA
   
        # total loss
        loss = loss_source + loss_target + lossDA
            
        # calculate metrics
        metrics = {}

        if stage:
           name = "_source_da_%s/" % self.da_loss_fnc_name if self.da_loss_fnc_name else "/" 
           metrics[f"filter/loss_total/{stage}"] = loss
           metrics[f"filter/recall{name}{stage}"] = self.source_recall(x_source, y_source)
           metrics[f"filter/precision{name}{stage}"] = self.source_precision(x_source, y_source)
           if self.da_loss_fnc_name: 
              name = "_da_%s/" % self.da_loss_fnc_name  
              metrics[f"filter/recall_target{name}{stage}"] = self.target_recall(x_target, y_target)
              metrics[f"filter/precision_target{name}{stage}"] = self.target_precision(x_target, y_target)
              metrics[f"filter/loss_source{name}{stage}"] = loss_source
              metrics[f"filter/loss_target{name}{stage}"] = loss_target
              if self.use_domain_adaptation:
                 metrics[f"filter/loss_temp_scaled{name}{stage}"] = lossDA
                 metrics[f"filter/loss_func{name}{stage}"] = loss_func 

        if stage == "train":
           name = "temp_source" if self.da_loss_fnc_name else "temperature" 
           metrics[f"filter/{name}"] = self.source_temp
           if self.da_loss_fnc_name:             
              metrics["filter/temp_target"] = self.target_temp
              metrics[f"filter/temp_da_{self.da_loss_fnc_name}"] = self.da_temp
                
        if stage in ["val", "test"]:
           self.source_cm_recall.update(x_source, y_source)
           self.source_cm_precision.update(x_source, y_source)
           if self.da_loss_fnc_name in self.domain_adaptation_classes:
              self.target_cm_recall.update(x_target, y_target)
              self.target_cm_precision.update(x_target, y_target)

        # run network and add output to graph object
        source_data["hit"].x_filter = source_data["hit"].x_filter.sigmoid()
        if self.da_loss_fnc_name in self.domain_adaptation_classes:
            target_data["hit"].x_filter = target_data["hit"].x_filter.sigmoid()
            
        if isinstance(data, Batch):
            # pylint: disable=protected-access
            data._slice_dict["hit"]["x_filter"] = data["hit"].ptr
            inc = torch.zeros(data.num_graphs, device=data["hit"].x.device)
            data._inc_dict["hit"]["x_filter"] = inc

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
            self.cm_logger.log(f"filter/recall_matrix_{stage}",
                               self.source_cm_recall, logger, epoch)
            self.cm_logger.log(f"filter/precision_matrix_{stage}",
                               self.source_cm_precision, logger, epoch)
        else:
            logger.experiment.add_figure(f"filter/recall_matrix_source_{stage}",
                          
                                     self.draw_confusion_matrix(self.source_cm_recall),
                                     global_step=epoch)
            self.source_cm_recall.reset()
            
            logger.experiment.add_figure(f"filter/recall_matrix_target_{stage}",
                                     self.draw_confusion_matrix(self.target_cm_recall),
                                     global_step=epoch)
            self.target_cm_recall.reset()
            
            logger.experiment.add_figure(f"filter/precision_matrix_source_{stage}",
                                self.draw_confusion_matrix(self.source_cm_precision),
                                global_step=epoch)
            self.source_cm_precision.reset()
            
            logger.experiment.add_figure(f"filter/precision_matrix_target_{stage}",
                                self.draw_confusion_matrix(self.target_cm_precision),
                                global_step=epoch)
            self.target_cm_precision.reset()