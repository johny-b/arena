# %%
try:
    import procgen_tools
except ModuleNotFoundError:
    commit = 'a6302f4'
    get_ipython().run_line_magic(magic_name='pip', line=f'install -U git+https://github.com/ulissemini/procgen-tools.git@{commit}')

from procgen_tools.utils import setup

setup() # create directory structure and download data

from procgen_tools.imports import widgets, display, HBox, plt, np, hook
from procgen_tools import visualization, patch_utils, maze, vfield

# %%
AX_SIZE = 4

checkbox = widgets.Checkbox(value=False, description='Show action components')

fig_out = widgets.Output()
text_out = widgets.Output()
display(text_out)
display(fig_out)

def generate_plots(max_size : int = 18, cols : int = 2, rows : int = 1):
    """ Generate rows*cols plots for random seeds with inner grid size at most max_size. """
    assert max_size >= 3, f"The smallest mazes have size 3; {max_size} is too small!"

    # Indicate that the plots are being generated
    with text_out:
        print(f'Generating {rows*cols} plots with max inner grid size {max_size}...')

    fig, axs = plt.subplots(rows, cols, figsize=(AX_SIZE*cols, AX_SIZE*rows))
    for idx, ax in enumerate(axs.flatten()):
        seed = np.random.randint(0, 100000)
        while maze.get_inner_grid_from_seed(seed=seed).shape[0] > max_size:
            seed = np.random.randint(0, 100000)
        venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
        vf = vfield.vector_field(venv, policy=hook.network)
        vfield.plot_vf(vf, ax=ax, show_components=checkbox.value)
        ax.set_title(f'Seed: {seed:,}')
        ax.axis('off')

    text_out.clear_output()
    fig_out.clear_output(wait=True)
    with fig_out:
        plt.show()


# Make slider for max inner grid size
slider = widgets.IntSlider(min=3, max=25, step=1, value=18, description='Max grid size')
display(slider)

# Make a button to generate new plots
button = widgets.Button(description='Generate new plots')
button.on_click(lambda _: generate_plots(max_size = slider.value))
display(HBox([button, checkbox]))

generate_plots(max_size = slider.value)
# %%
