This repositiory accompanies the paper "Resolving social dilemmas with minimal reward transfer", containing the code required to reproduce its results.

## Installation

Install the dependencies using miniconda

```shell
conda create -f environment.yml
```

Activate the environment and add the repo to the PYTHONPATH

```shell
conda activate rtm
export PYTHONPATH=$(pwd)
```

## Results

```shell
python3 src/rtm/solver_lp.py
```
