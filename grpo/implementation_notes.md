## Documenting some of the assumption I made to port PPO over to a non-language learning environment.

To map the LLMs to our simple environment (cartpole), I treated the questions, $q$ as the states, and the observations $o$ stemming from $q$ as paths starting from the state $q$.

For GRPO, I sampled a "main" path that represented the actions the 

## Some notes for the paper
This version of GRPO would only work for environments in which it is pretty inexpensive to explore the state space. For every observation, we need to sample $G$ paths of length $m$ from the environment to get their expected reward. This can become expensive and the group size and path lengths increase.


## Method - Group Relative Policy Optimization
Group Relative Policy Optimization (GRPO) is a modification to PPO in which the advantages are calculated using relative group averages rather than learning a neural network to approximate the value function [reference]. GRPO has been shown to be effective in language modelling environments, but there has not been much research on how this objective performs in non-language settings. For this paper, we developed a version of GRPO for a simple control environment with a discrete state space. We optimize the following objective,
$J_{GRPO}(\theta) = \mathbb E[\tau\sim\pi_\theta, \{\}]$
where

(does this essentially just become Monte Carlo?)


## Discussion - GRPO
Our modified version of GRPO 