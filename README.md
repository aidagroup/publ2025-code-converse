# Reinforcement Learning Optimizer Comparison Study

![alt text]([http://url/to/img.png](https://github.com/aidagroup/publ2025-code-converse/blob/main/GFX/rewards_plot.png))

## 📋 Overview
This repository contains code and analysis for comparing RAdam vs Adam optimizers in PPO reinforcement learning agent training. The study uses dynamically generated control systems with fixed seed reproducibility while allowing neural network weight initialization variations.

## 🚀 Quick Start

### Installation
``` bash
python -m venv rl_opt_env
source rl_opt_env/bin/activate  # Linux/MacOS
# or .\rl_opt_env\Scripts\activate for Windows

pip install -r requirements.txt
```
## Experiment Execution
To reproduce the experiments with fixed system generation seeds while allowing neural network weight initialization variability:

### Run the full experiment (160 runs: 80 RAdam + 80 Adam)
```python main.py```

### Visualize results using TensorBoard
```tensorboard --logdir=multi_opt_full_experiment/ --port 8888```


## Immedieatly Analyse our existing data
we already have saved all 160 CSVs in a file named "Combined_CSVs" and exstract the final rewards then perform one of the latter tests.

### Generate statistical analysis plots
python tensorboard_boxplot.py  # Comparative performance distributions
python tensorboard_fscore.py   # F-test significance analysis
python tensorboard_hist.py     # Reward distribution histograms
python tensorboard_plots.py    # Training progression curves
