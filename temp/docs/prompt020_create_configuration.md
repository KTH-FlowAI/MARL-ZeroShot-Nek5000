# Systematically setup the configuration for Nek-Solo usage 

# Background
## Usage
We have built the nek-solo mode which embeds the DRL policy into F77 and run the pure nek. This also support uniform blowing and opposition control. Now, I wanted to create new scripts for using nek-solo mode to run the evaluation in Channel cases.

So I migrated all of the configurations in HPC here in `./conf` folder. The channel cases are named as `omegaX_XXXX`. I have tested those in `conf/omega1_BASE/*embedded*` and need you to systematically extend this to the rest of cases. 

## Case descriptions
We consider the following cases: baseline (base), opposition (oc), dr-reward policy (dr) and nes-reward policy (nes-org). Belows I clarify the usage and some points worth tarcking your attention:

### Baseline: 
For baseline (*base*), I use uniform blowing control with 0 intensity to recover the wall boundary while maintain the framework usage. 
1. For this I make the `ndrl` be as the same as DRL used in this case. As we are not really interested in 
2. Then I use an arbitary `rec_freq` for sampling the binary `drlrec`. this can be `rec_freq=1` for the DRL case or maybe just 10 times higher is totally acceptable. 
3. The `nb_interactions` now it should be at least double of that of training (fixed `2500`). So maybe `6000` is a good one if the `n_drl` is identical to DRL
4. The `simulation.smpfreq` should be consistent with `n_drl` for generating the time-series. 
5. There is a parameter for generating `int_pos` with multiple y-dir planes which is used to the spectra analysis. 
```
y_planes: [0,3,5,8,10,12,15,20,23,26,30,40,50,75,100,150,200]
```
There the value represents the inner-scaled (i.e. wall-units) y-plane distance from the wall. This means it is bounded by the `simulation.retau` for each case. If you encounter higher `retau` please add more value for planes that further from the wall. And you can also interpolate value in between but not make it too fine. And this should be consistent with all the cases within one case folder. 

### DRL Case (include nes and dr): 
This is where we should use the embeded Nek-solo. 
1. Please always use the `runner.policy=best_model` 
2. Keep consistent `n_drl` with the training configuration 
3. The `nb_interactions` now it should be at least double of that of training (fixed `2500`). So maybe `6000` is a good one.
4. The same setup requirement as described in Baseline for time-series and `int_pos` writting

### OC case 
1. The `simulation.n_drl = 1` MUST be fulfilled as it is determined by the control law. 
2. Because of that, the sampling frequency of `drlrec` should be `embedded.rec_freq = n_drl * freq` where `n_drl` should be consistent with the DRL config. 
3. The `simulation.smpfreq=n_drl` where `n_drl` should be the one used in DRL config. 

## Warning 
1. DO NOT change any training configs that I put there so far!

# Objective:
1. Please create embedding usage cases for each channel cases and make sure they meet the requriement for each case as aforementioned. 
2. The source code folder should be consistent with the existing config in the folder.
3. Something I let you decided:
    + Creat a sub folder to store the embedded config or put them in parallel? 
    + For DRL evaluation, should we create a new run folder or put them in `eval` sub folder for the training run?
    + Can we shorten the `.yml` name to make the cases clearer and distinguishable? 
 