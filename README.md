# afl-bench
Test-bench to test aggregation strategies for asynchronous federated learning

# Development installation
1. Create and import a conda environconda with required dependencies: `conda env create -f env.yaml --name <YOUR_ENV_NAME>`
2. Activate new conda environment: `conda activate <YOUR_ENV_NAME>`
3. Install this repo package as a local module symbolically (assuming `cwd` is this repository): `pip install -e .`Set up `wandb` via running `wandb login`.

# TODOs
1. Should an agent pull immediately from the global model if it has trained the same version? I think not?
2. Supporting non-IID data-loaders and sampling.
