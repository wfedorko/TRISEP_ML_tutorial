# TRISEP_ML_tutorial

This repository holds the notebooks and code for the Machine Learning tutorial at TRISEP 2019 on August 1st and 2nd.
Before proceeding please fork this repository by clicking on a button above.


## Starting up on AWS instance
Log into your instance. Then launch a screen/tmux session. Next clone your repository, set up pytorch environment and launch jupyter notebook server. Instructions on how to set up ssh tunnel and bring up the jupyter root screen will be printed on your terminal.
```
screen
git clone <your forked repo url> TRISEP_ML_tutorial
. anaconda3/bin/activate pytorch_p36
cd TRISEP_ML_tutorial
. find_this_ip
./start_jupyternotebook.sh
```


