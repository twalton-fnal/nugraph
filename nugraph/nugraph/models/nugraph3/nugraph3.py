"""NuGraph3 model architecture"""
import argparse
import warnings
import psutil

import torch.cuda
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.data import Batch

from pytorch_lightning import LightningModule

from .types import Data
from .transform import Transform
from .encoder import Encoder
from .core import NuGraphCore
from .decoders import ( VertexDecoder, InstanceDecoder, FilterDecoder, SpacepointDecoder,
                        EventDecoder, SemanticDecoder )

from ...data import H5DataModule

class NuGraph3(LightningModule):
    """
    NuGraph3 model architecture, which includes the domain adaptation framework performed on 
    both the source and target data.
    
    Domain adaptation is available in both the event and semantic decoders. 
    Labels are used from both the source and target data for all event and semantic decoder losses.

    Args:
        in_features: Number of input node features
        hit_features: Number of hit node features
        nexus_features: Number of nexus node features
        interaction_features: Number of interaction node features
        instance_features: Number of instance features
        planes: Tuple of detector plane names
        semantic_classes: Tuple of semantic classes
        event_classes: Tuple of event classes
        num_iters: Number of message-passing iterations
        event_head: Whether to enable the event decoder
        semantic_head: Whether to enable the semantic decoder
        filter_head: Whether to enable filter decoder
        vertex_head: Whether to enable the vertex decoder
        instance_head: Whether to enable instance decoder
        spacepoint_head: Whether to enable the spacepoint decoder
        use_checkpointing: Whether to use checkpointing
        lr: Learning rate
        da_loss_fnc_name: Name of the domain adaptation loss function
                          (dann, mmd, semantic, sinkhorn)
        warmup_epochs: Enable a warmup period for increasing the learning rate
    """
    def __init__(self,
                 in_features: int = 4,
                 hit_features: int = 128,
                 nexus_features: int = 32,
                 interaction_features: int = 32,
                 instance_features: int = 8,
                 planes: tuple[str] = ("u","v","y"),
                 semantic_classes: tuple[str] = ('MIP','HIP','shower','michel','diffuse'),
                 event_classes: tuple[str] = ('numu','nue','nc'),
                 num_iters: int = 5,
                 event_head: bool = False,
                 semantic_head: bool = True,
                 filter_head: bool = True,
                 vertex_head: bool = False,
                 instance_head: bool = False,
                 spacepoint_head: bool = False,
                 use_checkpointing: bool = False,
                 lr: float = 0.001,
                 da_loss_fnc_name: str = None,
                 warmup_epochs: int = 0):
        super().__init__()

        warnings.filterwarnings("ignore", ".*NaN values found in confusion matrix.*")

        self.save_hyperparameters()

        self.nexus_features = nexus_features
        self.interaction_features = interaction_features

        self.semantic_classes = semantic_classes
        self.event_classes = event_classes
        self.num_iters = num_iters
        self.lr = lr
        self.warmup_epochs = warmup_epochs

        self.da_loss_fnc_name = da_loss_fnc_name
        self.domain_adaptation_classes = [ "dann", "mmd", "semantic", "sinkhorn" ]

        # encoder 
        self.encoder = Encoder(in_features, hit_features, nexus_features, interaction_features, 
                               instance_features=instance_features)

        # message-passing core
        self.core_net = NuGraphCore(hit_features, nexus_features, interaction_features,
                                    instance_features=instance_features, 
                                    use_checkpointing=use_checkpointing)
        
        # decoder functionalities
        self.decoders = []
        
        if event_head:
            self.event_decoder = EventDecoder(interaction_features, event_classes, 
                                              da_loss_fnc_name=self.da_loss_fnc_name, 
                                              warmup_epochs=self.warmup_epochs)
            self.decoders.append(self.event_decoder)

        if semantic_head:
            self.semantic_decoder = SemanticDecoder(hit_features, semantic_classes, 
                                                    da_loss_fnc_name=self.da_loss_fnc_name, 
                                                    warmup_epochs=self.warmup_epochs)
            self.decoders.append(self.semantic_decoder)

        if filter_head:
            self.filter_decoder = FilterDecoder(hit_features,)
            self.decoders.append(self.filter_decoder)

        if vertex_head:
            self.vertex_decoder = VertexDecoder(interaction_features)
            self.decoders.append(self.vertex_decoder)

        if instance_head:
            self.instance_decoder = InstanceDecoder(hit_features, instance_features)
            self.decoders.append(self.instance_decoder)

        if spacepoint_head:
            self.spacepoint_decoder = SpacepointDecoder(hit_features, len(planes))
            self.decoders.append(self.spacepoint_decoder)

        if not self.decoders:
            raise RuntimeError('At least one decoder head must be enabled!')

        # metrics
        self.max_mem_cpu = 0.
        self.max_mem_gpu = 0.    

    def forward(self, data: Data | list[Data], stage: str = None): # pylint: disable=arguments-differ
        """
        NuGraph3 forward function

        This function runs the forward pass of the NuGraph3 architecture,
        and then loops over each decoder to compute the loss and calculate
        and log any performance metrics.

        Args:
            data: Graph data object
            stage: String tag defining the step type
        """
        
        # Check if the input is a list of two batches
        batchA = data
        if self.da_loss_fnc_name in self.domain_adaptation_classes:
           if isinstance(data, list) and len(data) == 2:
              batchA, batchB = data
           else:
              raise ValueError("Expected input data to be a list of two batches.")

        # run the encoder and message-passing blocks
        if self.da_loss_fnc_name in self.domain_adaptation_classes:
           self.encoder(data=[batchA,batchB])
           for _ in range(self.num_iters):
               self.core_net(batchA)
               self.core_net(batchB)
        else:
           self.encoder(batchA)
           for _ in range(self.num_iters):
               self.core_net(batchA)

        # determine if the DA loss function is enabled for a decoder
        enable_da_decoders = []
        for name, decoder in {"event_decoder": getattr(self, "event_decoder", None),
                              "semantic_decoder": getattr(self, "semantic_decoder", None)
                             }.items():
            if decoder and hasattr(decoder, "use_domain_adaptation"):
               enable_da_decoders.append( decoder )

        # run the decoders and calculate the loss and metrics
        total_loss = 0.
        total_metrics = {}
        
        for decoder in self.decoders:
            if len(data) == 2: 
               if decoder in enable_da_decoders:
                  loss, metrics = decoder(data=[batchA, batchB], stage=stage)
               else:
                   raise ValueError("The decoder [", decoder, "] is not enabled.")
            else:
               loss, metrics = decoder(data=batchA, stage=stage)
            total_loss += loss
            total_metrics.update(metrics)
                   
        return total_loss, total_metrics

    
    def on_train_epoch_start(self) -> None:
        if self.da_loss_fnc_name == None:
           print("\nEnter training epoch")
        else:
           """ Check and toggle DA for event_decoder and semantic_decoder"""  
           epoch = self.trainer.current_epoch
           for name, decoder in {"event_decoder": getattr(self, "event_decoder", None),
                                 "semantic_decoder": getattr(self, "semantic_decoder", None)
                                }.items():
               if decoder is None or not hasattr(decoder, "use_domain_adaptation"):
                  continue

               if epoch < getattr(decoder, "warmup_epochs", 0):
                  print(f"[Epoch {epoch}] DA is OFF for {name} (warmup phase)")
                  continue

               if not decoder.use_domain_adaptation:
                  print(f"[Epoch {epoch}] Enabling DA for {name}") 
                  decoder.use_domain_adaptation = True

            
    def on_train_epoch_end(self) -> None:
        # stop updating running average for feature norm
        self.encoder.input_norms.update = False
        print("\n Finished training epoch")        

        
    def training_step(self,
                      batch: Data | list[Data],
                      batch_idx: int) -> float:
        loss, metrics = self(batch, 'train')
        if isinstance(batch, list) and len(batch) == 2:
            batchA, batchB = batch
        else:
            batchA = batch
        self.log('loss/train', loss, batch_size=batchA.num_graphs, prog_bar=True)
        self.log_dict(metrics, batch_size=batchA.num_graphs)
        self.log_memory(batch, 'train')
        return loss

    def on_validation_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch + 1
        print("\n Validation epoch ended!")
        for decoder in self.decoders:
            decoder.on_epoch_end(self.logger, 'val', epoch)    
    
    def validation_step(self,
                        batch: Data | list[Data],
                        batch_idx: int) -> None:
        loss, metrics = self(batch, 'val')
        if isinstance(batch, list) and len(batch) == 2:
            batchA, batchB = batch
        else:
            batchA = batch
        self.log('loss/val', loss, batch_size=batchA.num_graphs)
        self.log_dict(metrics, batch_size=batchA.num_graphs)

    def on_test_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch + 1
        for decoder in self.decoders:
            decoder.on_epoch_end(self.logger, 'test', epoch)
   
    def test_step(self,
                  batch: Data | list[Data],
                  batch_idx: int = 0) -> None:
        loss, metrics = self(batch, 'test')
        if isinstance(batch, list) and len(batch) == 2:
            batchA, batchB = batch
        else:
            batchA = batch
        self.log('loss/test', loss, batch_size=batchA.num_graphs)
        self.log_dict(metrics, batch_size=batchA.num_graphs)
        self.log_memory(batch, 'test')
        
    def predict_step(self,
                     batch,
                     batch_idx: int = 0) -> list[Data]:
        self(batch)
        return batch    

    def configure_optimizers(self) -> tuple:
        optimizer = AdamW(self.parameters(),
                          lr=self.lr)
        """
        onecycle = OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], {'scheduler': onecycle, 'interval': 'step'}
        """
        
        return optimizer

    def on_after_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Clamps eta_s, eta_t, eta_da after each optimizer step."""
        self.event_decoder.eta_s.data.clamp_(min=1e-3, max = 1)
        self.event_decoder.eta_t.data.clamp_(min=1e-3, max = 1)
        self.event_decoder.eta_da.data.clamp_(min=1e-3, max = 1)
        
    def log_memory(self, batch: Data | list[Data], stage: str) -> None:
        """
        Log CPU and GPU memory usage

        Args:
            batch: Data object to step over
            stage: String tag defining the step type
        """
        # get the one dataset
        if isinstance(batch, list) and len(batch) == 2:
            batchA, batchB = batch
        else:
            batchA = batch
        
        # log CPU memory
        if not hasattr(self, 'max_mem_cpu'):
            self.max_mem_cpu = 0.
        cpu_mem = psutil.Process().memory_info().rss / float(1073741824)
        self.max_mem_cpu = max(self.max_mem_cpu, cpu_mem)
        self.log(f'memory_cpu/{stage}', self.max_mem_cpu,
                 batch_size=batchA.num_graphs, reduce_fx=torch.max)

        # log GPU memory
        if not hasattr(self, 'max_mem_gpu'):
            self.max_mem_gpu = 0.
        if self.device != torch.device('cpu'):
            gpu_mem = torch.cuda.memory_reserved(self.device)
            gpu_mem = float(gpu_mem) / float(1073741824)
            self.max_mem_gpu = max(self.max_mem_gpu, gpu_mem)
            self.log(f'memory_gpu/{stage}', self.max_mem_gpu,
                     batch_size=batchA.num_graphs, reduce_fx=torch.max)
        
    @staticmethod
    def transform(planes: tuple[str]) -> Transform:
        """
        Return data transform for NuGraph3 model
        
        Args:
            planes: tuple of detector plane names
        """
        return Transform(planes)

    @staticmethod
    def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add argparse argument group for NuGraph3 model

        Args:
            parser: Argument parser to append argument group to
        """
        model = parser.add_argument_group('model', 'NuGraph3 model configuration')
        model.add_argument('--num-iters', type=int, default=5,
                           help='Number of message-passing iterations')
        model.add_argument('--in-feats', type=int, default=5,
                           help='Number of input node features')
        model.add_argument('--hit-feats', type=int, default=128,
                           help='Hidden dimensionality of hit convolutions')
        model.add_argument('--nexus-feats', type=int, default=32,
                           help='Hidden dimensionality of nexus convolutions')
        model.add_argument('--interaction-feats', type=int, default=32,
                           help='Hidden dimensionality of interaction layer')
        model.add_argument('--instance-feats', type=int, default=8,
                           help='Hidden dimensionality of object condensation')
        model.add_argument('--event', action='store_true',
                           help='Enable event classification head')
        model.add_argument('--semantic', action='store_true',
                           help='Enable semantic segmentation head')
        model.add_argument('--filter', action='store_true',
                           help='Enable background filter head')
        model.add_argument('--instance', action='store_true',
                           help='Enable instance segmentation head')
        model.add_argument('--vertex', action='store_true',
                           help='Enable vertex regression head')
        model.add_argument("--spacepoint", action="store_true",
                           help="Enable spacepoint prediction head")
        model.add_argument('--no-checkpointing', action='store_false',
                           dest="use_checkpointing",
                           help='Disable checkpointing during training')
        model.add_argument('--epochs', type=int, default=80,
                           help='Maximum number of epochs to train for')
        model.add_argument('--learning-rate', type=float, default=0.001,
                           help='Max learning rate during training')
        model.add_argument('--da-loss', type=str, choices=['dann', 'mmd', 'sinkhorn', 'semantic'], default=None,
                           help='Select the DA loss function to use (options are: dann, mmd, sinkhorn, semantic)')   
        model.add_argument('--warmup', type=int, default=0,
                           help='Include the warmup phase before the domain adaptation loss turns on. (default: %(default)s).')
        return parser

    @classmethod
    def from_args(cls, args: argparse.Namespace, nudata: H5DataModule) -> 'NuGraph3':
        """
        Construct model from arguments

        Args:
            args: Namespace containing parsed arguments
            nudata: Data module
        """
        return cls(
            in_features=args.in_feats,
            hit_features=args.hit_feats,
            nexus_features=args.nexus_feats,
            interaction_features=args.interaction_feats,
            instance_features=args.instance_feats,
            planes=nudata.planes,
            semantic_classes=nudata.semantic_classes,
            event_classes=nudata.event_classes,
            num_iters=args.num_iters,
            event_head=args.event,
            semantic_head=args.semantic,
            filter_head=args.filter,
            vertex_head=args.vertex,
            instance_head=args.instance,
            spacepoint_head=args.spacepoint,
            use_checkpointing=args.use_checkpointing,
            lr=args.learning_rate,
            da_loss=args.da_loss,
            warmup_epochs=args.warmup)
