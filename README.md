# fuzzy-mococo

Fuzzy MoCoCo experiments from the paper "A genetic fuzzy system for interpretable and parsimonious reinforcement learning policies" (https://doi.org/10.1145/3449726.3463198)

Most important file is the run script:

```./fuzzy_mococo.py```

this being the script that actually runs Fuzzy MoCoCo on Mountain Car.

Incidental scripts to pass args to this .py file and run on Slurm are:
```./fuzzy_mococo.sh``` and ```./run_fuzzy_mococo.sh```

## Dependencies for run script
rlenvs: https://github.com/jtbish/rlenvs

zadeh: https://github.com/jtbish/zadeh
