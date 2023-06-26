# %%
import numpy as np
import torch as t
from torch import nn
import os

from procgen_tools.imports import load_model
from procgen_tools import maze, visualization
from procgen_tools.models import human_readable_action, human_readable_actions, MAZE_ACTION_DELTAS
from procgen_tools.rollout_utils import rollout_video_clip, get_predict
device = t.device("cuda")

MAIN = __name__ == '__main__'

# %%

policy, hook = load_model()

# %%
MAZE_SIZE = 11

all_squares = [(x, y) for x in range(MAZE_SIZE) for y in range(MAZE_SIZE)]
even_odd_squares = [square for square in all_squares if not (square[0] % 2) and (square[1] % 2)]
odd_even_squares = [square for square in all_squares if (square[0] % 2) and not (square[1] % 2)]

# %%
def get_seed(maze_size: int) -> int:
    """Returns a random seed that creates maze of a given size."""
    while True:
        seed = np.random.randint(0, 100000000)
        if maze.get_inner_grid_from_seed(seed=seed).shape[0] == maze_size:
            return seed

def get_activations_sum(seed, layer_name, positions):
    """Returns mean activation absolute value per channel, for given layer_name and mouse positions"""
    venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
    venv_all, (legal_mouse_positions, _) = maze.venv_with_all_mouse_positions(venv)
    with t.no_grad():
        hook.run_with_input(venv_all.reset().astype('float32'))
        
    raw_activations = hook.values_by_label[layer_name]
    assert len(raw_activations.shape) == 4, "This layer has wrong shape"
    activations = raw_activations.abs().sum(dim=-1).sum(dim=-1)

    data = []
    for mouse_pos, activation in zip(legal_mouse_positions, activations):
        if mouse_pos in positions:
            data.append(activation)
            
    data = t.stack(data)
    data = data.mean(dim=0)
    
    return data

# %%

seed = get_seed(MAZE_SIZE)
layer_name = "embedder.relu3_out"
even_odd = get_activations_sum(seed, layer_name, even_odd_squares)
odd_even = get_activations_sum(seed, layer_name, odd_even_squares)
diff = (even_odd - odd_even)

TOPK_N = 30
even_odd_high = diff.topk(TOPK_N)
odd_even_high = diff.topk(TOPK_N, largest=False)
print(f"(Even, odd) squares have high values in channels {even_odd_high.indices.tolist()}, diff {even_odd_high.values.round().tolist()}")
print(f"(Odd, even) squares have high values in channels {odd_even_high.indices.tolist()}, diff {odd_even_high.values.round().tolist()}")


# %%
# TEST 2. Try ablations.
class ModelWithRelu3Ablations(nn.Module):
    def __init__(self, orig_policy: nn.Module, ablated_channels):
        super().__init__()
        self.orig_policy = orig_policy
        self.ablated_channels = tuple(ablated_channels)
    
    def forward(self, x):
        hidden = self.hidden(x)
        
        #   NOTE: everything below is just copied from procgen_tools.models.CategoricalPolicy
        from torch.distributions import Categorical
        import torch.nn.functional as F
        
        logits = self.orig_policy.fc_policy(hidden)
        log_probs = F.log_softmax(logits, dim=1)                                
        p = Categorical(logits=log_probs)                                       
        v = self.orig_policy.fc_value(hidden).reshape(-1)                                   
        return p, v
        
    def hidden(self, x):
        embedder = self.orig_policy.embedder
        x = embedder.block1(x)
        x = embedder.block2(x)
        x = embedder.block3(x)
        x = embedder.relu3(x)
        x = self._ablate_relu3(x)
        x = embedder.flatten(x)
        x = embedder.fc(x)
        x = embedder.relufc(x)
        return x
    
    def _ablate_relu3(self, x):
        x[:, self.ablated_channels] = 0
        return x

def assert_same_model_wo_ablations():        
    policy_with_ablations = ModelWithRelu3Ablations(policy, [])
    venv = maze.create_venv(num=1, start_level=get_seed(9), num_levels=1)
    obs = t.from_numpy(venv.reset()).to(t.float32)   

    with t.no_grad():
        categorical_0, value_0 = policy(obs)
        categorical_1, value_1 = policy_with_ablations(obs)
    assert t.allclose(categorical_0.logits, categorical_1.logits)
    assert t.allclose(value_0, value_1)

     
if MAIN:        
    assert_same_model_wo_ablations()
        
# %%

#   Channels that are important for (even, odd) fields
even_odd_channels = even_odd_high.indices   # (73, 110, 112, 6, 96)
odd_even_channels = odd_even_high.indices  # (121, 17, 20, 123, 45)
policy_ablated_even_odd = ModelWithRelu3Ablations(policy, even_odd_channels)
policy_ablated_odd_even = ModelWithRelu3Ablations(policy, odd_even_channels)

venv = maze.create_venv(num=1, start_level=get_seed(MAZE_SIZE), num_levels=1)
obs = t.from_numpy(venv.reset()).to(t.float32)   

with t.no_grad():
    categorical_0, value_0 = policy(obs)
    categorical_1, value_1 = policy_ablated_even_odd(obs)

state = maze.state_from_venv(venv)

#  %% 
# TEST ON (1, 0)
state.set_mouse_pos(1, 0)
new_venv = maze.venv_from_grid(state.inner_grid())

obs = t.from_numpy(new_venv.reset()).to(t.float32)   

with t.no_grad():
    categorical_0, value_0 = policy(obs)
    categorical_1, value_1 = policy_ablated_even_odd(obs)
    categorical_2, value_2 = policy_ablated_odd_even(obs)

print("ORIGINAL")
print(categorical_0.logits)
print("EVEN ODD ABLATED")
print(categorical_1.logits)
print("ODD EVEN ABLATED")
print(categorical_2.logits)

# %%
# TEST ON (0, 1)
state.set_mouse_pos(0, 1)
new_venv = maze.venv_from_grid(state.inner_grid())

obs = t.from_numpy(new_venv.reset()).to(t.float32)   


with t.no_grad():
    categorical_0, value_0 = policy(obs)
    categorical_1, value_1 = policy_ablated_even_odd(obs)
    categorical_2, value_2 = policy_ablated_odd_even(obs)

print("ORIGINAL")
print(categorical_0.logits)
print("EVEN ODD ABLATED")
print(categorical_1.logits)
print("ODD EVEN ABLATED")
print(categorical_2.logits)
# %%

# seed = get_seed(MAZE_SIZE)

venv_1 = maze.create_venv(num=1, start_level=seed, num_levels=1)
venv_2 = maze.create_venv(num=1, start_level=seed, num_levels=1)
venv_3 = maze.create_venv(num=1, start_level=seed, num_levels=1)


orig_vector_field = visualization.vector_field(venv_1, policy)
ablate_even_odd_vector_field = visualization.vector_field(venv_2, policy_ablated_even_odd)
ablate_odd_even_vector_field = visualization.vector_field(venv_2, policy_ablated_odd_even)

# %%
visualization.plot_vfs(orig_vector_field, ablate_even_odd_vector_field)
# %%
visualization.plot_vfs(orig_vector_field, ablate_odd_even_vector_field)
# %%

def get_vf(seed, policy):
    venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
    return visualization.vector_field(venv, policy)

# %%
#   CLEAN SUMMARY OF THE ABOVE

seed = get_seed(MAZE_SIZE)
layer_name = "embedder.relu3_out"
even_odd = get_activations_sum(seed, layer_name, even_odd_squares)
odd_even = get_activations_sum(seed, layer_name, odd_even_squares)
diff = (even_odd - odd_even)

# %%
TOPK_N = 30
even_odd_high = diff.topk(TOPK_N)
odd_even_high = diff.topk(TOPK_N, largest=False)
print(f"(Even, odd) squares have high values in channels {even_odd_high.indices.tolist()}, diff {even_odd_high.values.round().tolist()}")
print(f"(Odd, even) squares have high values in channels {odd_even_high.indices.tolist()}, diff {odd_even_high.values.round().tolist()}")

# %%

print(odd_even_high.indices)

# %%
even_odd_channels = even_odd_high.indices   # (73, 110, 112, 6, 96)
odd_even_channels = odd_even_high.indices  # (121, 17, 20, 123, 45)

policy_ablated_even_odd = ModelWithRelu3Ablations(policy, even_odd_channels)
policy_ablated_odd_even = ModelWithRelu3Ablations(policy, odd_even_channels)

vf_original = get_vf(seed, policy)
vf_even_odd = get_vf(seed, policy_ablated_even_odd)
vf_odd_even = get_vf(seed, policy_ablated_odd_even)

visualization.plot_vfs(vf_original, vf_even_odd)
visualization.plot_vfs(vf_original, vf_odd_even)

# %%
