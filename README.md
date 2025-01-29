# go4robo

Work for my MITsdm Masters' Thesis.

## Project Setup

### Conda Environment

[Install Conda locally](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), navigate to this directory, and run:

`conda env create --name blenv --file conda_env.yml`

_Note to self: to export the conda environment, activate it and then run: `conda env export --file conda_env.yml --no-builds`_

### Blender

Install [Blender (Version 4.3.2) via the website](https://www.blender.org/download/).

Then point blender to the Conda Environment using these steps:
    1. In Blender’s installation folder, rename the “python” directory (e.g., to “python_backup”).
    2. Navigate to Blender’s installation folder and link the Conda environment to "python". On Linux/Mac the command is: `ln -s python /Users/robosquiggles/miniforge3/envs/blenv`

### VSCode

I followed [this YouTube tutorial by CG Python](https://www.youtube.com/watch?v=_0srGXAzBZE) to get VSCode set up with Blender.
