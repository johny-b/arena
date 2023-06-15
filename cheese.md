## TRAINING THE MOUSE ##

Q: Why do I want to train the main model?  
A: Solution should work whatever was the initial training seed --> we should probably try multiple models.

Q: What are all these repos?  
A: There is the well-known [OpenAI repo](https://github.com/openai/procgen) with many small procedurally created games.
We use a modified version of this repo (so that we have modified games) and a repo that contains training scripts.

Q: How long will this take?  
A: IDK. Current guess: 10h or so.

### Installation

1. Prereqs

Python 3.8, we'll be using torch 1.13.1.

2. Install https://github.com/JacobPfau/procgenAISC

```
# Install qty
sudo apt install -y qtcreator qtbase5-dev qt5-qmake cmake

# Install procgenAISC (NOTE: installation instructions have a wrong repo URL, I guess?)
git clone git@github.com:JacobPfau/procgenAISC.git
cd procgenAISC
pip3 install -e .

# This should say "building procgen...done".
# Note that "maze_aisc" is the environment from this repo (and probably the main we care about?)
python -c "from procgen import ProcgenGym3Env; ProcgenGym3Env(num=1, env_name='maze_aisc')"

# Play an episode (will not work on remote without any special tricks):
python -m procgen.interactive --env-name maze_aisc
```

3. Install dependencies

```
pip3 install torch==1.13.1 pyyaml pandas tensorboard==2.5
```

4. Get the training repo and run a script
```
git clone git@github.com:jbkjr/train-procgen-pytorch.git
cd train-procgen-pytorch

# NOTE: this is just an example from the repo, hopefully the one we're interested in (I don't know what is variant 1/2)
python train.py --exp_name maze1 --env_name maze_aisc --num_levels 100000 --distribution_mode hard --param_name hard-500 --num_timesteps 200000000 --num_checkpoints 5 --seed 1080
```
