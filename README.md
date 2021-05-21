# Multi-Armed Bandits
This repository contains a notebook that visualizes different Multi-Armed Bandit algorithms in the context of online advertising. You can play a game and manually optimize a virtual campaign. Your results are compared to the different MAB algorithms. 

## Content
You can find an interactive explanation of the most popular Multi-Armed Bandit algorithms (:math:`\epsilon`-greedy, Thompson Sampling, Upper Confidence Bounds) in the notebook `multi-armed-bandit-algorithms.ipynb`. In `optimization_simulator.ipynb`, you can manually optimize a campaign and compete with the implemented algorithms. `beta-distribution-plotter.ipynb` contains an interactive plotter based on the `bokeh` library to visualize the $Beta$ distribution, which is the conjugate prior to the Binomial distribution. It visualizes the impact on clicks & bounces on the posterior distribution over click rates. 
