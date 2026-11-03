"""Weighting of the domain adaptation (DA) loss term"""

import math
import torch


DA_WEIGHT_MODES = ("schedule", "fixed", "learned")


def dann_ramp(progress: float, gamma: float = 10.0) -> float:
  """
  DANN ramp from Ganin & Lempitsky (2015), https://arxiv.org/abs/1409.7495

  lambda(p) = 2 / (1 + exp(-gamma * p)) - 1, rising smoothly from 0 at p = 0
  to ~1 at p = 1. With gamma = 10 it reaches ~0.9 at 30% of training.

  Args:
      progress: Fraction of DA training completed, clipped to [0, 1]
      gamma: Steepness of the ramp
  """
  p = min(max(progress, 0.0), 1.0)
  return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


class DAWeight:
  """
  Decides how strongly the DA term enters the total loss.

  Modes:
      schedule: lambda = max_weight * dann_ramp(progress, gamma)   (default)
      fixed:    lambda = max_weight
      learned:  the original behaviour, 2 * exp(-t) * L + t with a trainable t.
                Kept so earlier results can be reproduced and ablated.

  For an adversarial loss (DANN), lambda is applied inside the gradient
  reversal layer, which is the canonical DANN formulation: the domain
  classifier always trains at full strength, and only the reversed gradient
  reaching the feature extractor is scaled by lambda. For the non-adversarial
  losses (MMD, Sinkhorn, semantic alignment), lambda multiplies the loss.

  This is a plain object rather than an nn.Module, so it adds nothing to the
  model's state_dict and existing checkpoints load unchanged.

  Args:
      mode: One of DA_WEIGHT_MODES
      max_weight: Final (schedule) or constant (fixed) DA weight
      gamma: Steepness of the DANN ramp in schedule mode
  """
  def __init__(self, mode: str = "schedule", max_weight: float = 1.0,
               gamma: float = 10.0):
      if mode not in DA_WEIGHT_MODES:
          raise ValueError(f"da_weight_mode must be one of {DA_WEIGHT_MODES}, got \"{mode}\"")
      if max_weight < 0:
          raise ValueError(f"da_weight must be non-negative, got {max_weight}")
      self.mode = mode
      self.max_weight = max_weight
      self.gamma = gamma
      # fraction of DA training completed; updated by the model each training batch
      self.progress = 0.0

  @property
  def learned(self) -> bool:
      """Whether the DA weight is the trainable temperature"""
      return self.mode == "learned"

  @property
  def value(self) -> float:
      """Current lambda (1.0 in learned mode, where the temperature does the weighting)"""
      if self.mode == "fixed":
          return self.max_weight
      if self.mode == "schedule":
          return self.max_weight * dann_ramp(self.progress, self.gamma)
      return 1.0

  @property
  def grl_alpha(self) -> float:
      """Scale for the gradient reversal layer"""
      return self.value

  def __call__(self, loss_func: torch.Tensor, temperature: torch.Tensor,
               adversarial: bool) -> torch.Tensor:
      """
      Combine the raw DA loss into the term added to the total loss

      Args:
          loss_func: Raw DA loss (domain classifier cross-entropy, MMD, ...)
          temperature: The decoder's DA temperature parameter (learned mode only)
          adversarial: True when lambda has already been applied in the
                       gradient reversal layer (DANN)
      """
      if self.learned:
          return 2 * (-1 * temperature).exp() * loss_func + temperature
      if adversarial:
          return loss_func
      return self.value * loss_func
