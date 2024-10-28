This repository accompanies the paper "Resolving social dilemmas with minimal reward transfer", [Springer Nature](https://link.springer.com/article/10.1007/s10458-024-09675-4), containing the code required to reproduce its results.

A related paper that uses Multi-agent reinforcement learning and focuses on Markov games can be found at this [repo](https://github.com/willis-richard/meltingpot/).

## Installation

Install the dependencies using conda

```shell
conda create -f environment.yml
```

Activate the environment and add the repo to the PYTHONPATH

```shell
conda activate rtm
export PYTHONPATH="$(pwd)/src"
```

## Algorithms

src.rtm.algorithms contains Algorithm 1 (called find_T_star), which returns a minimal reward transfer matrix for a normal-form social dilemma. It also has a simpler algorithm to find the symmetrical self-interest level of a game, and algorithm to maximise the off-diagonal entropy of a reward transfer matrix.

## Results

In order to produce the results from the paper, call

```shell
python3 src/rtm/solver_lp.py --game <GAME_NAME>
```

To see the options, use

```shell
python3 src/rtm/solver_lp.py --help
```
