# Group17-dvr
This is repository for CS492 development project - Differentiable Volumetric Rendering [paper](https://arxiv.org/abs/1912.07372).
We borrowed marching cube code for occupancy network ("/utils" and "setup.py") from [Occupancy network github]( https://github.com/autonomousvision/occupancy_networks/tree/master/im2mesh).
# Setup
type 
```
conda env create --file environment.yaml
conda activate DVR
python setup.py build_ext --inplace
```
to create conda environment and compile "/utils". You may adjust pytorch version based on your environment.
# Dataset
You can download DTU MVS dataset from [Here](https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/DTU.zip).
create "data" folder in the main directory and place the dataset there.
# Generate output
You can download pretrained model [Here](https://drive.google.com/file/d/1bkQ7XoDMz1832BIcKrON1nTpKhCtEWZw/view?usp=sharing).
Unzip the out.zip and place in main directory.
Then, type
```
python generate.py config/(whatever)/(youwant).yaml
```
to generate output mesh. For example, if you want to generate mesh for 'skull' scene trained with mvs depth supervision, type
```
python generate.py config/skull/mvs.yaml
```
# Train
If you want to train the model from scratch, type
```
python fit.py config/(whatever)/(youwant).yaml
```
# Evaluation
We used official DTU MVS dataset evaluation code [Here](https://roboimagedata.compute.dtu.dk/?page_id=36) 'surface' mode.
Download 'SampleSet' and run the matlab code with generated mesh.

