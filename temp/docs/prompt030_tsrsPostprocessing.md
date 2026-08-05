# Postprocessing of the TimeSeries data in a Jupyter notebook 

## Background: 
@post_processing/postlib/tsrs_case.py @post_processing/tsrs_stitch.py  @post_processing/postlib/tsrs_spectra.py @post_processing/tsrs_eval.py  I am at the stage of post-processing the spectra and snapshots from the cases in `./runs/`. My idea is to have a jupyter notebook like @post_processing/deterministic.ipynb or @post_processing/drlrec_cases.ipynb that can process several cases and overlay the results together within one notebook scripts to have visualization of data and figures. 

## Design 
My design is 
1. Let's focus on the `env_001` data by default instead of including all of `env_001` to `env_006` data as we do not need them here. 
2. We have a dictionary to manage the file name and path so they can be processed one by one. The code should contain `tsrs_stitch.py` function that stitches the `pts` data and convert the stacked data into a `.npy` file. 
3. Then I need visualization of the wall plane (`y=0`) and the sensing plane (`y+=15`) visulaization for velocity fluctuation and pressure fluctuation, this fluctuation field is obtained by subtracting the mean in time dimension for each single point.

Plus, I need the the snapshots across all avaliable time steps are saved in the results folder. Please separate the variable by variable. 
For saving folder, take reference from @post_processing/drlrec_cases.ipynb and @utils/collect-results.

And we can take a quick visualization of the u and v fields at an arbitary step for all included cases. 

4. [Optional] We can use the snapshots to make gifs to be visualized for each case and each quantity.  

5. Post-process the spectra data. I am specifically interested in:
    1. `map` (Power-spectrum density, func of y+ and \lambda_z for uu+ and -uv+) 
    2. The two-point correlation (at y^+ 15 should be prioritized).

We should make a quick visualization for the PSD and two-point correlation. PSD take one figure and two-point correlation takes another one. Each case is put into a subfigure.

6. The processed spectra data should be saved into the same folder where store the time-series snapshots too. I belive @post_processing/tsrs_spectra.py should have the code to look at it

7. Finally archive the stitched raw `.npz` data into the case result folder too so we can verify if we need to overwrite or not next time. 