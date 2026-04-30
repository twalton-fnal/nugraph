## Nugraph Workspace Setup

### Instructions for multiple development workspaces 

1. Log in to the Elastic facility analysis
1. Select a node deploying a 20 or 40 GPU device
1. Update the **bash_profile.sh** file with:
    - export LD_LIBRARY_PATH=\\$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
1. Create nested conda environments using **--stack**
    - This enables creating a based conda environment supporting multiple Nugraph installations
    1. conda create --name=NugraphBase # the Fermilab EAF deploys python 3.10.20
    2. conda env update --name NugraphBase --file \<Path to yaml file>/env_base.yaml
    3. conda activate NugraphBase
    4. pip install torch-scatter --no-build-isolation
1. Create the nested conda environments 
    1. conda create --name NugraphDA python=3.10.20
    2. conda create --name NugraphMain python=3.10.20
1. Activate the **NugraphDA** or **NugraphMain** nested conda environments   
    1. conda activate --stack NugraphDA
    2. Check the compatibility between the torch package and the system NVIDIA driver.
       1. nvidia-smi
       2. python -c "import sys; sys.path.append('/home/\\${USER}/.conda/envs/NugraphBase/lib/python3.10/site-packages'); import torch; print(torch.__version__); print(torch.version.cuda)"
    1. If the **Base** and **DA** versions are incompatible, continue with the following steps:
        1. pip uninstall torch
        2. pip install torch --index-url https://download.pytorch.org/whl/cu126
1. Install the kernel for the selected conda environment  
    1. conda install ipykernel
    2. python -m ipykernel install --user --name NugraphDA --display-name "Python (NugraphDA)"
1. Go to the **NugraphDA** workspace and install the **nugraph da** package
    1. cd /home/twalton/NuGraphGPUWorkspace/
    2. git clone https://github.com/AleksCipri/NuGraph3DA.git
    3. cd NuGraph3DA
    4. git checkout -b 14-labeled-target-all-decoders
    5. cd nugraph
    6. pip install --no-deps -e .
    7. If installing the official nugraph repository (git@github.com:nugraph/nugraph.git):
         1. cd pynuml
         2. pip install --no-deps -e .

  


  
   
  
  