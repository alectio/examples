# Project 1: Manual Curation

## Setting up Local Environment

The basic requirement for running this experiment is having python running on your machine. We suggest you create a new virtual env for running the experiment.

```bash
conda create -n alectio_env python=3.9
conda activate alectio_env
```

Now let us install the sdk and other requirements:

First navigate to the manual_curation_demo folder and follow the commands given below.

```bash
cd /path/to/manual_curation_demo
pip install alectio_sdk
pip install -r requirements.txt
```

## Steps to Start the Experiment

To run this experiment the user must follow the following steps:

1. Login to Alectio Portal
2. Create a new project with data source as Alectio Public Dataset (BrainTumor).
3. Inside the project create a new experiment with QS = Manual Curation.
4. Now click on the run button and select some samples from the GUI and then click on start training.

## Running the Experiment on Local machine

If your machine doesn't have a GPU, modify the code inside processes.py all the lines that have GPU related code have comments about required modifications for CPU. Once your environment is set up and code has been modified copy the experiment token ```(Alectio->Projects->Experiments->get_token)``` and copy it to ```main.py``` and ```processes.py``` files.

The next step is to run the main.py file.

```bash
python3 main.py
```

Congratulations! you have now started an experiment sucessfully.

After completion of each loop the experiment will be paused the user now has to go to the Alectio Portal and select the records for the next loop and click on start training.
