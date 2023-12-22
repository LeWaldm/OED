
This codebase implements abstract classes and corresponding functions for Bayesian Optimal Experimental Design (BOED) in PyTorch.

### Features
- BOED over discrete candidate designs with Nested Monte Carlo (NMC) Expected Information Gain (EIG) computations
- BOED over continuous design space by maximizing Prior Contrastive Estimation (PCE) (https://arxiv.org/abs/1911.00294)
- BOED by Deep Adaptive Design (DAD) (https://arxiv.org/abs/2103.02438)
- variational inference algorithm

### Files in src
- `data_utils`: abtract Experimenter class for modelling experimental settings, abstract Design_Network class for DAD
- `distributions`: abstract prior and likelihood classes
- `model_utils`: functions for discrete BOED, continuous BOED with PCE, and DAD building on abstract Experimenter and distribution classes

### Examples
We demonstrate the codebase on 2 different versions of 2d tissue slicing closely aligned to https://github.com/andrewcharlesjones/spatial-experimental-design: 
- discrete: iteration over finite candidate designs with NMC
- continuous: weighting each point on the grid by its distance to the chosen slice and optimize PCE

Additionally, there is an experimental example with a simple 1d location finding inspired by https://arxiv.org/abs/2103.02438 and a 2d tissue model that samples by interpolation from an affine grid.

### Additional Features
- constraining distribution params by simple transformations during optimization

### Note of Caution
- Testing was limited (only performed on one example).
- Eventually, other variance reduction methods for the gradient estimation need to be implemented.