# Flash Flood Recovery Analysis
This repository contains the codes and supplementary materials for the paper [10.1175/JHM-D-23-0196.1](https://doi.org/10.1175/JHM-D-23-0196.1). 

<figure>
    <img src="https://github.com/user-attachments/assets/325c0f9b-988c-4eb2-8992-1880bfcb6ab1" alt="image" width="639">
    <figcaption>High Resolution Recoveriness Map</figcaption>
</figure>


## Overview
We have developed and implemented a new metric called Recoveriness to estimate the recovery potential of watersheds after flash floods. Using 78 years of historical flood data and advanced machine learning techniques, we provide probabilistic estimates of flash flood recoveriness across the conterminous United States. This approach models the recession limb of the hydrograph, which is essential in understanding post-flood recovery but has been less studied compared to the rising limb.

## Key Contributions:
- Introduced Recoveriness, a metric for assessing watershed recovery potential.
- Identified significant geomorphological and climatological predictors, including slope index, river basin area, and river length.
- Mapped localized hotspots for flash flood recoveriness, highlighting areas in Kentucky, Tennessee, West Virginia, western Montana, and northern Idaho.
- Utilized Quantile Random Forest to provide probabilistic recovery estimates.

## Usage:
- You can proceed by either cloning the repo or downloading the zipped version of the repo. 
- Download all data from [zenodo](https://doi.org/10.5281/zenodo.13729992). Put the downloaded data into the 'data' folder of the repo.
- Run the program in sequence starting from '01-prepare_data.py'

## Published Dataset
- [Recoveriness.csv](https://zenodo.org/records/13729469)
