"""NuGraph3 semantic decoder"""
from typing import Any
import tempfile
import matplotlib.pyplot as plt
import seaborn as sn

import torch
import torchmetrics as tm
from torch import nn
from torch_geometric.data import Batch
from pytorch_lightning.loggers import Logger

from ....util import ConfusionMatrixLogger, RecallLoss
from ....util.DANNLoss import ReverseLayerF
from ....util.MMDLoss import MMDLoss
from ....util.SemanticLoss import SemanticAlignmentLoss
from ....util.SinkhornLoss import Sinkhorn
from ....util.EmbeddingPlotter import CombinedEmbeddingPlot
from ..types import Data

class SemanticDecoder(nn.Module):
    """
    NuGraph3 semantic decoder module

    Convolves the planar node embedding down to a set of categorical scores for
    each semantic class.

    The implemented DA loss functions are:
    - Domain-Adversarial Neural Networks (DANN)
    - Maximum Mean Discrepancy (MMD)
    - Semantic alignement loss (semantic)

    Args:
        hit_features: Number of planar hit node features
        semantic_classes: List of semantic classes
        
    """
    def __init__(self,
                 hit_features: int,
                 semantic_classes: list[str],
                 da_loss_fnc_name: str = None,
                 warmup_epochs: int = 0):
        super().__init__()

        self.semantic_classes = semantic_classes
        self.warmup_epochs = warmup_epochs
        self.da_loss_fnc_name = da_loss_fnc_name
        self.use_domain_adaptation = False 
        self.domain_adaptation_classes = [ "dann", "mmd", "semantic", "sinkhorn" ]
        
        # loss function
        self.loss = RecallLoss()
        if self.da_loss_fnc_name == "dann":
           self.loss_dann = nn.CrossEntropyLoss()  
           loss_dann = torch.tensor(0.0)    
        elif self.da_loss_fnc_name == "mmd":
           self.loss_mmd = MMDLoss()  
           loss_mmd = torch.tensor(0.0)
        elif self.da_loss_fnc_name == "semantic":
           semantic_metric = "cosine" #or metric='euclidean' 
           self.loss_semantic = SemanticAlignmentLoss(semantic_metric)
           semantic = torch.tensor(0.0) 
        elif self.da_loss_fnc_name == "sinkhorn":
           self.loss_sinkhorn = Sinkhorn(blur=0.05)
           sinkhorn = torch.tensor(0.0) 
        
        # temperature parameter
        self.source_temp = nn.Parameter(torch.tensor(0.))
        if self.da_loss_fnc_name in self.domain_adaptation_classes: 
           self.target_temp = nn.Parameter(torch.tensor(0.))
           self.da_temp = nn.Parameter(torch.tensor(0.))

        # metrics
        metric_args = {
            "task": "multiclass",
            "num_classes": len(semantic_classes),
            "ignore_index": -1
        }
        
        self.source_recall = tm.Recall(**metric_args)
        self.source_precision = tm.Precision(**metric_args)
        self.source_cm_recall = tm.ConfusionMatrix(normalize="true", **metric_args)
        self.source_cm_precision = tm.ConfusionMatrix(normalize="pred", **metric_args)
        self.cm_logger = ConfusionMatrixLogger(semantic_classes)

        if self.da_loss_fnc_name in self.domain_adaptation_classes:
           self.target_recall = tm.Recall(**metric_args)
           self.target_precision = tm.Precision(**metric_args)
           self.target_cm_recall = tm.ConfusionMatrix(normalize="true", **metric_args)
           self.target_cm_precision = tm.ConfusionMatrix(normalize="pred", **metric_args)
           self.embeddings = CombinedEmbeddingPlot(method="umap")

        # network
        self.net = nn.Linear(hit_features, len(semantic_classes))
            
        self.classes = semantic_classes

        # Domain classifier network for DANN
        if self.da_loss_fnc_name == "dann":
           self.domain_net = nn.Sequential(
                                  nn.Linear(in_features=5,out_features=64),
                                  nn.ReLU(),
                                  nn.Linear(in_features=64,out_features=2)
           )  
           self.domain_classes = ['source', 'target']

        

    
    def forward(self, data: Data | list[Data], stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 semantic decoder forward pass

        Args:
            data: Graph data object or both the source and target data objects
            stage: Stage name (train/val/test)
        """

        # the data
        if not self.da_loss_fnc_name in self.domain_adaptation_classes:
           source_data = data
        else:
           source_data, target_data = data 
            
        # run network and add output to graph object      
        def _run_net_and_add_output(data: Any, net: torch.nn.Module, key: str):
            data[key].x_semantic = net(data[key].x)
            if isinstance(data, Batch):
               data._slice_dict[key]["x_semantic"] = data[key].ptr
               inc = torch.zeros(data.num_graphs, device=data[key].x.device)
               data._inc_dict[key]["x_semantic"] = inc

        _run_net_and_add_output(source_data, self.net, "hit")
        if self.da_loss_fnc_name in self.domain_adaptation_classes:
           _run_net_and_add_output(target_data, self.net, "hit")

        # calculate loss
        loss = loss_source = loss_target = lossDA = raw_lossDA = 0
    
        x_source = source_data["hit"].x_semantic
        y_source = source_data["hit"].y_semantic
        w_source = 2 * (-1 * self.source_temp).exp()
        loss_source = w_source * self.loss(x_source, y_source) + self.source_temp
            
        if self.da_loss_fnc_name in self.domain_adaptation_classes:
           x_target = target_data["hit"].x_semantic
           y_target = target_data["hit"].y_semantic
           w_target = 2 * (-1 * self.target_temp).exp()
           loss_target = w_target * self.loss(x_target, y_target) + self.target_temp
            
        if self.da_loss_fnc_name == None:
           loss = loss_source

        """
            Domain Adaptation is optional and is implemented via DANN (lossDA). 
            Both source and target labels are being used to calculate 
            semantic losses (loss_source and loss_target).
            All losses are scaled by their own temperatures, which are trainable parameters.
            DA loss is also capped at most 1/4 of the loss_source (to be on the safe side). 
        """
        if self.use_domain_adaptation:
           wDA = 2 * (-1 * self.da_temp).exp()
            
           if self.da_loss_fnc_name == "dann":    
              alpha = 1
               
              reversed_S = ReverseLayerF.apply(source_data["hit"].x_semantic, alpha)
              xSS = self.domain_net(reversed_S)
              ySS = torch.zeros(xSS.shape[0], dtype=torch.long, device=xSS.device)
            
              reversed_T = ReverseLayerF.apply(target_data["hit"].x_semantic, alpha)
              xTT = self.domain_net(reversed_T) 
              yTT = torch.ones(xTT.shape[0], dtype=torch.long, device=xTT.device)    

              combined_image = torch.cat((xSS, xTT), dim=0)  
              combined_label = torch.cat((ySS, yTT), dim=0)

              wDA = 2 * (-1 * self.da_temp).exp()
              raw_lossDA = wDA * self.loss_dann(combined_image, combined_label) + self.da_temp

           elif self.da_loss_fnc_name == "mmd":
                raw_lossDA = wDA * self.loss_mmd(source_data["hit"].x_semantic, target_data["hit"].x_semantic) + self.da_temp
           elif self.da_loss_fnc_name == "semantic":
                raw_lossDA = self.loss_semantic(source_data["hit"].x_semantic, y_source, target_data["hit"].x_semantic, y_target) 
           elif self.da_loss_fnc_name == "sinkhorn":
                pairwise_distances = torch.cdist(source_data["hit"].x_semantic, target_data["hit"].x_semantic, p=2)
                flattened_distances = pairwise_distances.view(-1)
                max_distance = torch.max(flattened_distances)
                dynamic_blur_val = 0.05 * max_distance.detach().cpu().numpy()
                raw_lossDA = wDA * self.loss_sinkhorn(source_data["hit"].x_semantic, 
                                                      target_data["hit"].x_semantic, blur=max(dynamic_blur_val, 0.01)) + self.da_temp 
           else:
              sys.exit( f"The function {self.da_loss_fnc_name} does not exist." )
   

           """  
           Smooth capping of the DA based on the source semantic loss value 
           (currently to be at most 1/4 of the event loss value)
           sharpness of transition when source semantic loss goes from positive to negative
           """            
           sharp = 20.0  
           sig = torch.sigmoid(sharp * loss_source)
           max_lossDA = sig * (loss_source / 4) + (1 - sig) * (4 * loss_source)
           lossDA = torch.min(raw_lossDA, max_lossDA)
   
        # total loss
        loss = loss_source + loss_target + lossDA
            
        # calculate metrics
        metrics = {}

        name = "_source_%s/" % self.da_loss_fnc_name if self.da_loss_fnc_name else "/"
        if stage:
           metrics[f"loss_semantic{name}{stage}"] = loss
           metrics[f"recall_semantic{name}{stage}"] = self.source_recall(x_source, y_source)
           metrics[f"precision_semantic{name}{stage}"] = self.source_precision(x_source, y_source)
           if self.da_loss_fnc_name: 
              metrics[f"recall_semantic_target/{stage}"] = self.target_recall(x_target, y_target)
              metrics[f"precision_semantic_target/{stage}"] = self.target_precision(x_target, y_target)
              metrics[f"loss_semantic_source/{stage}"] = loss_source
              metrics[f"loss_semantic_target/{stage}"] = loss_target
              if self.da_loss_fnc_name == "dann":
                 metrics[f"DA_loss_capped_semantic/{stage}"] = lossDA
                 metrics[f"DA_loss_uncapped_semantic/{stage}"] = raw_lossDA 

        if stage == "train":
           metrics["temperature/semantic"] = self.source_temp            
           if self.da_loss_fnc_name in self.domain_adaptation_classes:
              metrics["temperature/semantic_target"] = self.target_temp
              metrics["temperature/semantic_DA"] = self.da_temp
                
        if stage in ["val", "test"]:
           self.source_cm_recall.update(x_source, y_source)
           self.source_cm_precision.update(x_source, y_source)
           if self.da_loss_fnc_name in self.domain_adaptation_classes:
              self.target_cm_recall.update(x_target, y_target)
              self.target_cm_precision.update(x_target, y_target)
              self.embeddings.update(source_data["hit"].x_semantic, y_source, target_data["hit"].x_semantic, y_target)

        # apply softmax to prediction
        source_data["hit"].x_semantic = source_data["hit"].x_semantic.softmax(dim=1)
        if self.da_loss_fnc_name in self.domain_adaptation_classes:
           target_data["hit"].x_semantic = target_data["hit"].x_semantic.softmax(dim=1)

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
        if self.da_loss_fnc_name in self.domain_adaptation_classes:
           plt.title(f"Domain Adaptation ({self.da_loss_fnc_name.upper()})")
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
        
        if not logger:
            return

        if self.da_loss_fnc_name == None:
           self.cm_logger.log(f"semantic/recall-matrix-{stage}",
                                self.source_cm_recall, logger, epoch)
           self.cm_logger.log(f"semantic/precision-matrix-{stage}",
                                self.source_cm_precision, logger, epoch)
        else:
           logger.experiment.add_figure(f"recall_semantic_matrix_source/{stage}",
                                     self.draw_confusion_matrix(self.source_cm_recall),
                                     global_step=epoch)
           self.source_cm_recall.reset()

           logger.experiment.add_figure(f"recall_semantic_matrix_target/{stage}",
                                     self.draw_confusion_matrix(self.target_cm_recall),
                                     global_step=epoch)
           self.target_cm_recall.reset()

           logger.experiment.add_figure(f"precision_semantic_matrix_source/{stage}",
                                self.draw_confusion_matrix(self.source_cm_precision),
                                global_step=epoch)
           self.source_cm_precision.reset()

           logger.experiment.add_figure(f"precision_semantic_matrix_target/{stage}",
                                self.draw_confusion_matrix(self.target_cm_precision),
                                global_step=epoch)
           self.target_cm_precision.reset()

           # Plot the embedding space 
           dat1, lab1, dat2, lab2 = self.embeddings.compute()
           dat1sub, lab1sub = self.embeddings.subsample(dat1, lab1, max_samples=1000)
           dat2sub, lab2sub = self.embeddings.subsample(dat2, lab2, max_samples=1000)
        
           embeddings_fig = self.embeddings.plot_combined(dat1sub, lab1sub, dat2sub, lab2sub, epoch=epoch, class_names=self.semantic_classes)
           logger.experiment.add_figure(f"Embeddings semantic/{stage}",
                                     embeddings_fig, global_step=epoch)
           self.embeddings.reset()
