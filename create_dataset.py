# %%
import numpy as np

from procgen_tools import maze, visualization

# %%
def visualize_maze(state: maze.EnvState) -> None:
    venv = maze.venv_from_grid(state.inner_grid())
    visualization.visualize_venv(venv, render_padding=False)    
    
def create_maze_with_decision_square(size: int) -> maze.EnvState:
    assert size % 2, "size must be an odd number"
    while True:
        seed = np.random.randint(0, 100000)
        while maze.get_inner_grid_from_seed(seed=seed).shape[0] != size:
            seed = np.random.randint(0, 100000)
        venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
        state_bytes = venv.env.callmethod("get_state")[0]
        
        if maze.maze_has_decision_square(state_bytes):
            return maze.state_from_venv(venv)

def put_mouse_on_decision_square(state: maze.EnvState) -> None:
    decision_square = maze.get_decision_square_from_maze_state(state)
    padding = maze.get_padding(state.inner_grid())
    state.set_mouse_pos(decision_square[0] + padding, decision_square[1] + padding)

state = create_maze_with_decision_square(9)
visualize_maze(state)
put_mouse_on_decision_square(state)
visualize_maze(state)


# decision_square = maze.get_decision_square_from_maze_state(state)
# print(decision_square)
# print(decision)
# %%
