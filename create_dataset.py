# %%
import json
from typing import Tuple
import numpy as np

from procgen_tools import maze, visualization

# %%
def visualize_maze(state: maze.EnvState) -> None:
    venv = maze.venv_from_grid(state.inner_grid())
    visualization.visualize_venv(venv, render_padding=False)    
    
def create_maze_with_decision_square(size: int) -> Tuple[int, maze.EnvState]:
    assert size % 2, "size must be an odd number"

    while True:
        seed = np.random.randint(100000000)
        while maze.get_inner_grid_from_seed(seed=seed).shape[0] != size:
            seed = np.random.randint(100000000)

        venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
        state_bytes = venv.env.callmethod("get_state")[0]
        
        if maze.maze_has_decision_square(state_bytes):
            return seed, maze.state_from_venv(venv)

def put_mouse_on_decision_square(state: maze.EnvState) -> None:
    decision_square = maze.get_decision_square_from_maze_state(state)
    padding = maze.get_padding(state.inner_grid())
    state.set_mouse_pos(decision_square[0] + padding, decision_square[1] + padding)

seed, state = create_maze_with_decision_square(19)
visualize_maze(state)
put_mouse_on_decision_square(state)
# visualize_maze(state)

# %%

def generate_maze(grid_size: int):
    seed, state = create_maze_with_decision_square(grid_size)
    put_mouse_on_decision_square(state)
    return seed, state.inner_grid().tolist()

data = []
seeds = set()

MAZE_COUNT = 1000
GRID_SIZE = 19

while len(data) < MAZE_COUNT:
    seed, grid = generate_maze(GRID_SIZE)
    if not any(any(x == maze.CHEESE for x in row) for row in grid):
        continue

    if seed not in seeds:
        data.append((seed, grid))
        seeds.add(seed)

from pprint import pprint
print(json.dumps(data))
# %%
