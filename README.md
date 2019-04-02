# MNASNet 1.0 trainer
Training setup for MNASNet that will get you close to "paper" numbers.

Result after 200 epochs:
  * Top-1 = 73.512
  * Top-5 = 91.544
  
This program with hardcoded settings requires 4x NVIDIA GTX 1080ti GPUs to run.

## Requirements:

  * PyTorch 1.0.1
  * PillowSIMD
  * TensorboardX

And this PR: https://github.com/pytorch/vision/pull/829

## Training log and checkpoint

See under `training_state`
