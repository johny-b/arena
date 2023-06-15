# LOOKING FOR CHEESE

## RESOURCES

1.  Entrypoint: [Understanding and controlling a maze-solving policy network](https://www.lesswrong.com/posts/cAC4AXiNC5ig6jQnc/understanding-and-controlling-a-maze-solving-policy-network)
2.  [Goal misgeneralization in deep reinforcment learning](https://arxiv.org/abs/2105.14111) - Training the maze-solving networks. 
This describes the network(s) we'll be working on. [Training section](#training-a-mouse) should more-or-less reproduce this article.
3.  https://github.com/ulissemini/procgen-tools - Repository used by Alex Turner et al. Few parts important at the first glance:
    
    *   [models.py](https://github.com/UlisseMini/procgen-tools/blob/main/procgen_tools/models.py) - Interpretable version of the models used in the original article.
        "Interpretable version" means (I'm not super-sure) "the same architecture, so we can load weights, but written in a little different way so that
        we can use [CircRL - Tools for applying circuits-style interpretability techniques to RL agents](https://github.com/UlisseMini/circrl).
        [ref](https://github.com/ulissemini/procgen-tools#interpretable-parameter-compatible-implementations-of-impala-models)

4.  [Google drive with trained models](https://drive.google.com/drive/folders/1Ig7bzRlieyYFcdKL_PM-guSWR8WryDOL). 
    
    * AT et al used `maze_I/*.pth `.
    * Maybe only `maze_I/model_rand_region_5.pth`? I don't recall any part about other model.

5.  [Cheese algebra colab](https://colab.research.google.com/drive/1fPfehQc1ydnYGSDXZmA22282FcgFpNTJ?usp=sharing)
6.  [Maze generator colab](https://colab.research.google.com/drive/1zHk6jxjTjQ4yL12Fbp3REpTXsqQGV1dp?usp=sharing) (I'm not sure if this is useful)
7.  [Channel resampling colab](https://colab.research.google.com/drive/1uAUc91NHdpMiJqjlwh6Lx1_32QNLki5Z?usp=sharing) (I'm not sure if this is useful)

## Questions

* [The cheese vector from seed A usually doesn't work on seed B](https://www.lesswrong.com/posts/cAC4AXiNC5ig6jQnc/understanding-and-controlling-a-maze-solving-policy-network#The_cheese_vector_from_seed_A_usually_doesn_t_work_on_seed_B) I'm not sure if I understand this fully. Does this have any important implications for what we want to do?
* Shards and stuff: does this theory matter for us?

## Training a mouse

Q: **Why do I want to train the main model?**  
A: Solution should work whatever was the initial training seed --> we should probably try multiple models.

Q: **What are all these repos?**  
A: There is the well-known [OpenAI repo](https://github.com/openai/procgen) with many small procedurally created games.
We use a modified version of this repo (so that we have modified games) and a repo that contains training scripts.

Q: **How long will this take?**  
A: IDK. Current guess: 20h or so. Article says `Each training run required approximately 30 GPU hours of compute on a V100`, we have `A10` instead of `V100` so should be faster.

1. Prereqs

    Python 3.8, we'll be using torch 1.13.1.

2. Install https://github.com/JacobPfau/procgenAISC

    ```
    # Install qty
    sudo apt install -y qtcreator qtbase5-dev qt5-qmake cmake
    
    # Install procgenAISC (NOTE: installation instructions in the procgenAISC repo have a wrong URL here, I guess?)
    # NOTE: procgen_tools has a (commented) line that installs git+https://github.com/UlisseMini/procgenAISC@lauro
    #       (--> not main, but some other branch). I don't know if this matters at all.
    git clone git@github.com:JacobPfau/procgenAISC.git
    cd procgenAISC
    pip3 install -e .
    
    # This should say "building procgen...done".
    # Note that "maze_aisc" is the environment from this repo (and probably the main we care about?)
    python -c "from procgen import ProcgenGym3Env; ProcgenGym3Env(num=1, env_name='maze_aisc')"
    
    # Play an episode (will not work on remote without some special tricks):
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
