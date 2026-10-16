# HPC IsaacLab 3.0 setup steps


Log in
```
@access.engr.oregonstate.edu
@submit.hpc.engr.oregonstate.edu
srun -A mime -p ampere --gres=gpu:1 --mem=40G --pty bash
```

Build the sandbox container
```
apptainer build --sandbox isaaclab-sandbox isaaclab-3.0.def
mv -v ~/hpc-share/apptainer/isaaclab-sandbox/IsaacLab ~/hpc-share/IsaacLab
ln -s ~/hpc-share/IsaacLab ~/hpc-share/apptainer/isaaclab-sandbox/IsaacLab
```

Run sandbox container
```
apptainer shell --nv isaaclab-sandbox
```
