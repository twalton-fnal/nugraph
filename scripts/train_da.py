#!/usr/bin/env python

"""
Train NuGraph3, with or without domain adaptation (DA).

A script version of notebooks/train-neutrino-data.ipynb: every notebook
setting is a command-line option whose default is the notebook's value, so
running it with no options does what the notebook does.

Examples
    # the notebook as committed: MicroBooNE -> ICARUS, DANN, 10 epochs, warmup 2
    python train_da.py

    # MMD for 30 epochs with the DANN-style ramp on the DA weight
    python train_da.py --da-loss mmd --epochs 30 --da-weight-mode schedule

    # paired source/target training without a DA loss (needs the "off" fix)
    python train_da.py --da-loss off

    # nominal NuGraph3 on one file
    python train_da.py --input-files 1 --source /path/to/icarus-numi-1d.gnn.h5 \\
        --dataset-name Nugraph-Icarus-Event-Decoder-Test

    # print the resolved configuration and stop
    python train_da.py --dry-run
"""

import argparse
import getpass
import os
import sys
from datetime import datetime
from pathlib import Path

# notebook defaults (cell 8)
DEFAULT_SOURCE = "/scratch/7DayLifetime/cerati/concat-final-makeup.tiny.gnn.h5"
DEFAULT_TARGET = "/scratch/7DayLifetime/cerati/icarus-numi-1d.gnn.h5"
DEFAULT_DATASET_NAME = "MicroBoone-Icarus-Small"
DEFAULT_EVENT_CLASSES = "cc_nue,cc_numu,cc_nutau,nc"
DA_LOSSES = ("off", "dann", "mmd", "semantic", "sinkhorn")


def configure(argv=None) -> argparse.Namespace:
    doc = __doc__.strip()
    p = argparse.ArgumentParser(description=doc.split("\n\n")[0],
                                epilog=doc.split("\n\n", 2)[2],
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    run = p.add_argument_group("run configuration (notebook cell 8)")
    run.add_argument("--input-files", type=int, choices=(1, 2), default=2,
                     help="1 = nominal NuGraph3 on the source file; 2 = DA model on source and "
                          "target (default %(default)s)")
    run.add_argument("--source", default=DEFAULT_SOURCE, help="source .gnn.h5 file")
    run.add_argument("--target", default=DEFAULT_TARGET, help="target .gnn.h5 file (input-files 2)")
    run.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME,
                     help="prefix of the TensorBoard run folder (default %(default)s)")
    run.add_argument("--epochs", type=int, default=10, help="training epochs (default %(default)s)")
    run.add_argument("--da-loss", choices=DA_LOSSES, default="dann",
                     help="DA loss for input-files 2 (default %(default)s)")
    run.add_argument("--no-domain-adaptation", dest="run_domain_adaptation", action="store_false",
                     help="never switch DA on: the warmup is set to the number of epochs, as the "
                          "notebook does with run_domain_adaptation = False")
    run.add_argument("--warmup", type=int, default=None,
                     help="epochs before DA switches on; default as in the notebook: 0 for one file, "
                          "2 for two, all epochs with --no-domain-adaptation")
    run.add_argument("--batch-size", type=int, default=64 * 2, help="graphs per batch (default %(default)s)")
    run.add_argument("--no-event", dest="event_head", action="store_false", help="disable the event decoder")
    run.add_argument("--no-semantic", dest="semantic_head", action="store_false",
                     help="disable the semantic decoder")
    run.add_argument("--no-filter", dest="filter_head", action="store_false", help="disable the filter decoder")

    data = p.add_argument_group("data module (notebook cells 10-12)")
    data.add_argument("--event-classes", default=DEFAULT_EVENT_CLASSES,
                      help="comma-separated event classes, set on the data module as in the notebook "
                           "(default %(default)s)")
    data.add_argument("--num-workers", type=int, default=5,
                      help="data loader worker processes (data module default %(default)s)")
    data.add_argument("--shuffle", choices=("random", "balance"), default="random",
                      help="batch shuffling (data module default %(default)s)")
    data.add_argument("--balance-frac", type=float, default=0.1,
                      help="fraction of largest graphs spread across batches with --shuffle balance "
                           "(default %(default)s)")

    model = p.add_argument_group("network (notebook cell 14)")
    model.add_argument("--in-feats", type=int, default=5)
    model.add_argument("--hit-feats", type=int, default=128)
    model.add_argument("--nexus-feats", type=int, default=32)
    model.add_argument("--instance-feats", type=int, default=32)
    model.add_argument("--interaction-feats", type=int, default=32)
    model.add_argument("--num-iters", type=int, default=5)
    model.add_argument("--lr", type=float, default=0.001, help="learning rate (default %(default)s)")
    model.add_argument("--no-checkpointing", dest="use_checkpointing", action="store_false",
                       help="disable activation checkpointing (on in the notebook)")
    model.add_argument("--da-weight-mode", choices=("learned", "schedule", "fixed"), default="learned",
                       help="DA loss weighting (model default %(default)s, which the notebook uses)")
    model.add_argument("--da-weight", type=float, default=1.0,
                       help="final (schedule) or constant (fixed) DA weight (default %(default)s)")
    model.add_argument("--da-gamma", type=float, default=10.0,
                       help="steepness of the schedule ramp (default %(default)s)")
    model.add_argument("--drop-target-label", action="store_true",
                       help="leave the target task losses out of the total loss")

    out = p.add_argument_group("logging and trainer (notebook cells 8, 16-19)")
    out.add_argument("--user", default=getpass.getuser(),
                     help="user in the log path (default: current user, %(default)s)")
    out.add_argument("--log-root", default=None,
                     help="log root; default /scratch/7DayLifetime/<user>/NuGraphLogs as in the notebook")
    out.add_argument("--device", type=int, default=0,
                     help="GPU index; a negative value trains on CPU (default %(default)s)")
    out.add_argument("--ckpt-path", default=None, help="checkpoint to resume training from")
    out.add_argument("--no-test", dest="run_test", action="store_false", help="skip trainer.test after fitting")
    out.add_argument("--dry-run", action="store_true", help="print the configuration and stop")

    args = p.parse_args(argv)
    if args.epochs < 1:
        p.error("--epochs must be at least 1")
    if args.warmup is not None and args.warmup < 0:
        p.error("--warmup cannot be negative")
    return args


def resolve(args: argparse.Namespace) -> dict:
    """The notebook's cell-8 logic: DA loss, warmup and log directory"""
    two_files = args.input_files == 2
    da_loss = args.da_loss if two_files else None
    if args.warmup is not None:
        warmup = args.warmup
    elif args.run_domain_adaptation:
        warmup = 2 if two_files else 0
    else:
        warmup = args.epochs
    log_root = Path(args.log_root) if args.log_root else \
        Path(f"/scratch/7DayLifetime/{args.user}/NuGraphLogs")
    if two_files:
        log_dir = log_root / f"NuGraph-DA-{args.da_loss.upper()}" / f"EPOCH{args.epochs}-WARMUP{warmup}"
    else:
        log_dir = log_root / "NuGraph" / f"EPOCH{args.epochs}-WARMUP{warmup}"
    event_classes = [c.strip() for c in args.event_classes.split(",") if c.strip()]
    return {"two_files": two_files, "da_loss": da_loss, "warmup": warmup,
            "log_dir": log_dir, "event_classes": event_classes,
            "folder_name": f"{args.dataset_name}-{datetime.now():%Y-%m-%d-%H}"}


def check_inputs(args, cfg) -> None:
    files = [args.source] + ([args.target] if cfg["two_files"] else [])
    missing = [f for f in files if not Path(f).is_file()]
    if missing:
        sys.exit("Input file(s) not found: " + ", ".join(missing))
    if cfg["two_files"] and Path(args.source).resolve() == Path(args.target).resolve():
        sys.exit("--source and --target are the same file.")
    if args.event_head and not cfg["event_classes"]:
        sys.exit("The event decoder needs --event-classes.")
    if cfg["warmup"] >= args.epochs and cfg["two_files"] and cfg["da_loss"] != "off":
        print(f"Note: warmup ({cfg['warmup']}) >= epochs ({args.epochs}), so the DA loss never switches on.")


def main(argv=None) -> int:
    args = configure(argv)
    cfg = resolve(args)

    if cfg["two_files"]:
        print("running nugraph3 DA model with loss function [", args.da_loss.upper(), "]")
    else:
        print("running nominal nugraph3 model")
    os.environ["NUGRAPH_LOG"] = str(cfg["log_dir"])
    print("Files are in the directory [", os.environ["NUGRAPH_LOG"], "]")
    print(f"source: {args.source}" + (f"\ntarget: {args.target}" if cfg["two_files"] else ""))
    print(f"epochs {args.epochs}, warmup {cfg['warmup']}, batch size {args.batch_size}, "
          f"decoders: event={args.event_head} semantic={args.semantic_head} filter={args.filter_head}")
    print(f"DA weighting: {args.da_weight_mode}"
          + (f" (weight {args.da_weight}, gamma {args.da_gamma})" if args.da_weight_mode != "learned" else "")
          + (", target task losses dropped" if args.drop_target_label else ""))
    print(f"run folder: {cfg['folder_name']}")
    if args.dry_run:
        return 0
    check_inputs(args, cfg)

    # heavy imports only once we really train (cell 6)
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger
    import nugraph as ng
    print(ng.__file__)

    # data module (cells 10-12)
    Data = ng.data.NuGraphDataModule
    Model = ng.models.NuGraph3
    data_args = dict(model=Model, batch_size=args.batch_size, num_workers=args.num_workers,
                     shuffle=args.shuffle, balance_frac=args.balance_frac,
                     data_source_path=args.source)
    if cfg["two_files"]:
        data_args["data_target_path"] = args.target
    nudata = Data(**data_args)
    nudata.event_classes = cfg["event_classes"]
    print(nudata)
    print(nudata.semantic_classes)
    print(nudata.event_classes)

    # network (cell 14)
    nugraph = Model(
        in_features=args.in_feats,
        hit_features=args.hit_feats,
        nexus_features=args.nexus_feats,
        instance_features=args.instance_feats,
        interaction_features=args.interaction_feats,
        semantic_classes=nudata.semantic_classes,
        event_classes=nudata.event_classes,
        num_iters=args.num_iters,
        event_head=args.event_head,
        semantic_head=args.semantic_head,
        filter_head=args.filter_head,
        vertex_head=False,
        instance_head=False,
        use_checkpointing=args.use_checkpointing,
        lr=args.lr,
        da_loss_fnc_name=cfg["da_loss"],
        da_weight_mode=args.da_weight_mode,
        da_weight=args.da_weight,
        da_gamma=args.da_gamma,
        drop_target_label=args.drop_target_label,
        warmup_epochs=cfg["warmup"])

    # logger and callbacks (cells 16-17)
    cfg["log_dir"].mkdir(parents=True, exist_ok=True)
    logger = TensorBoardLogger(save_dir=cfg["log_dir"], name=cfg["folder_name"])
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.ModelCheckpoint(monitor="loss/val", mode="min"),
    ]

    # trainer (cell 19)
    accelerator, devices = ng.util.configure_device(args.device if args.device >= 0 else None)
    print(accelerator)
    print(devices)
    trainer = pl.Trainer(accelerator=accelerator,
                         devices=devices,
                         max_epochs=args.epochs,
                         logger=logger,
                         callbacks=callbacks)
    trainer.fit(nugraph, datamodule=nudata, ckpt_path=args.ckpt_path)
    if args.run_test:
        trainer.test(datamodule=nudata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
