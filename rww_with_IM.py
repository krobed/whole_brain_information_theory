
import os
import time
cpu = True
if cpu:
    N = 8
    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={N}'

# Import all required libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import jax
import jax.numpy as jnp
import copy
import optax
from scipy import io

# Jax enable x64
jax.config.update("jax_enable_x64", True)

# Import from tvboptim
from tvboptim.types import Parameter, Space, GridAxis
from tvboptim.types.stateutils import show_parameters
from tvboptim.utils import set_cache_path, cache
from tvboptim.execution import ParallelExecution, SequentialExecution
from tvboptim.optim.optax import OptaxOptimizer
from tvboptim.optim.callbacks import MultiCallback, DefaultPrintCallback, SavingCallback

# Network dynamics imports
from tvboptim.experimental.network_dynamics import Network, solve, prepare
from tvboptim.experimental.network_dynamics.dynamics.tvb import ReducedWongWang
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling, FastLinearCoupling
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph, DenseGraph
from tvboptim.experimental.network_dynamics.solvers import Heun
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.data import load_structural_connectivity, load_functional_connectivity

# BOLD monitoring
from tvboptim.observations.tvb_monitors.bold import Bold

# Observation functions
from tvboptim.observations.observation import compute_fc, fc_corr, rmse

# Set cache path for tvboptim
set_cache_path("./rww")
jax.config.update("jax_enable_x64", True)

import sys
import urllib
import zipfile
import argparse

args = argparse.ArgumentParser()
args.add_argument('-measure', type=str ,default='fc')
args.add_argument('-order', type=int ,default=2) # Order of the information measure. Note that for functional connectivity (correaltion) and PhiID computations order=2, whereas for HOI order>=3.

if __name__ == '__main__':
    # Parse arguments
    args = args.parse_args
    measure = args.measure
    order = args.order



    # --- Detect environment ---
    if "google.colab" in sys.modules:
        base_path = "/content/cmha_data"
    else:
        base_path = os.path.join(os.getcwd(), "cmha_data")

    os.makedirs(base_path, exist_ok=True)

    url = "https://zenodo.org/records/10572216/files/cmha-1.0.0.zip?download=1"
    zip_path = os.path.join(base_path, "cmha-1.0.0.zip")
    expected_folder = os.path.join(base_path, "cmha-1.0.0")

    # --- Check if already extracted ---
    if os.path.exists(expected_folder):
        print(f"Dataset already extracted at: {expected_folder}")
    else:
        # --- Download if missing ---
        if not os.path.exists(zip_path):
            print(f"Downloading dataset from Zenodo to {zip_path}...")
            urllib.request.urlretrieve(url, zip_path)
            print("Download complete.")
        else:
            print("Zip file already exists, skipping download.")

        # --- Extract ---
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(base_path)
        print("\nData extracted at:", base_path)

    from tvb.datatypes.connectivity import Connectivity


    labels = np.genfromtxt('cmha_data/cmha-1.0.0/data/CoCoMac/labels.tsv', dtype=str, delimiter='\t', usecols=1,skip_header=1)

    # Setup the Connectivity object
    conn = Connectivity()

    # Assign the required attributes
    weights = np.loadtxt('Connectivity2/weights.txt')[:82,:82]
    conn.weights = weights/np.max(weights)
    conn.tract_lengths = np.loadtxt('Connectivity2/tract_lengths.txt')[:82,:82]
    conn.region_labels = labels
    n_nodes = weights.shape[0]
    # Assign optional but recommended attributes to avoid 'None' warnings
    # For the 82-region Macaque RM atlas, if you don't have coordinates:
    conn.centres = np.zeros((len(labels), 3)) 
    conn.cortical = np.ones(len(labels), dtype=bool) # Most CoCoMac sets are cortical

    # This will now internalize the weights and calculate 'number_of_regions'
    conn.configure()


    import pandas as pd
    # Load record information
    record_metadata = pd.read_csv(f"{base_path}/cmha-1.0.0/data/participants.tsv",delimiter='\t')
    # Load fMRI records
    data_CoCoMAC = np.load(f"{base_path}/cmha-1.0.0/data/CoCoMac/timeseries.npy")

    print("Subjects were recorded under 6 states",record_metadata['cond'].unique())
    record_metadata.head(5)
    print("\n")
    print(f"Data shape {data_CoCoMAC.shape} indicates that there are 156 records from 82 brain regions with 500 samples each")
    all_states = {}

    for state_name in record_metadata['cond'].unique():
        all_states[state_name] = {}
        state_df = record_metadata.query("cond==@state_name")
        sub_idx = 1
        for idx, row in state_df.iterrows():
            all_states[row.cond][sub_idx] = data_CoCoMAC[idx]
            sub_idx+=1



    for state in all_states:
        emp_fc = np.mean([np.corrcoef(all_states[state][n].T) for n in all_states[state]], axis=0)
        fig, ax1 = plt.subplots(1, figsize=(8.1, 4.05), sharey=True)
        im1 = ax1.imshow(emp_fc, cmap="cividis", vmax=0.5)
        ax1.set_title(f"Functional Connectivity {state} state")
        ax1.set_xlabel("Region")
        ax1.set_ylabel("Region")
        cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.74, label="Connection Strength [a.u.]", extend='max')

    model = ReducedWongWang()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.1, 4.05), sharey=True)
    im1 = ax1.imshow(conn.weights, cmap="cividis", vmax=1)
    ax1.set_title("Structural Connectivity")
    ax1.set_xlabel("Region")
    ax1.set_ylabel("Region")
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.74, label="Connection Strength [a.u.]", extend='max')

    im2 = ax2.imshow(conn.tract_lengths, cmap="cividis")
    ax2.set_title("Tract Lengths")
    ax2.set_xlabel("Region")
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.74, label="Tract Length [mm]")

    plt.tight_layout()
    graph = DenseGraph(conn.weights, region_labels=conn.region_labels)
    dynamics = ReducedWongWang(w=0.3, I_o=0.32, INITIAL_STATE=(0.3,))
    coupling = FastLinearCoupling(local_states=["S"], G=0.15)
    noise = AdditiveNoise(sigma=0.00283, apply_to="S")

    # Assemble the network
    network = Network(
        dynamics=dynamics,
        coupling={'instant': coupling},
        graph=graph,
        noise=noise
    )


    t1 = 90_000  # Total simulation duration (ms) - 2 minutes
    dt = 4.0      # Integration timestep (ms)
    model, state = prepare(network, Heun(), t1=t1, dt=dt)

    result_init = model(state)


    network.update_history(result_init)
    model, state = prepare(network, Heun(), t1=t1, dt=dt)


    result = model(state)

    bold_monitor = Bold(
        period=1000.0,          # BOLD sampling period (1 TR = 1000 ms)
        downsample_period=4.0,  # Intermediate downsampling matches dt
        voi=0,                  # Monitor first state variable (S)
        history=result_init     # Use initial state as warm start
    )


    bold_result = bold_monitor(result)


    from matplotlib.colors import Normalize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.1, 3.0375))

    # Plot raw neural activity (first 1000 ms)
    t_max_idx = int(1000 / dt)
    time_raw = result.time[:t_max_idx]
    data_raw = result.data[:t_max_idx, 0, :]

    num_lines = data_raw.shape[1]
    cmap = plt.cm.cividis
    mean_values = np.mean(data_raw, axis=0)
    norm = Normalize(vmin=np.min(mean_values), vmax=np.max(mean_values))
    for i in range(num_lines):
        color = cmap(norm(mean_values[i]))
        ax1.plot(time_raw, data_raw[:, i], color=color, linewidth=0.5)

    ax1.text(0.95, 0.95, "Raw Neural Activity", transform=ax1.transAxes, fontsize=10,
            ha='right', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("S [a.u.]")

    # Plot BOLD signal (first 60 TRs)
    t_bold_max = 60
    time_bold = bold_result.time[:t_bold_max]
    data_bold = bold_result.data[:t_bold_max, 0, :]

    num_lines = data_bold.shape[1]
    mean_values = np.mean(data_bold, axis=0)
    norm = Normalize(vmin=np.min(mean_values), vmax=np.max(mean_values))
    for i in range(num_lines):
        color = cmap(norm(mean_values[i]))
        ax2.plot(time_bold, data_bold[:, i], color=color, linewidth=0.8)

    ax2.text(0.95, 0.95, "BOLD Signal", transform=ax2.transAxes, fontsize=10,
            ha='right', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("BOLD [a.u.]")

    plt.tight_layout()


    fc_target = np.corrcoef(all_states['awake'][1].T)
    def observation(state):
        """Compute functional connectivity from simulated BOLD signal."""
        # Run simulation
        result = model(state)
        # Convert to BOLD
        bold = bold_monitor(result)
        # Compute FC, skipping first 20 TRs to avoid transient effects
        fc = compute_info_measure(bold, skip_t=20)
        return fc

    def loss(state):
        """Compute RMSE between simulated and empirical FC."""
        fc = observation(state)
        return rmse(fc, fc_target)


    # Calculate initial FC
    fc_initial = np.array(observation(state))

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.1, 3.54375))

    # Plot both FC matrices
    for ax_current, fc_matrix, title_prefix in zip([ax1, ax2], [fc_target, fc_initial], ["Target FC", "Initial FC"]):
        fc_matrix = np.copy(fc_matrix)
        np.fill_diagonal(fc_matrix, np.nan)  # Set diagonal to NaN
        im = ax_current.imshow(fc_matrix, cmap='cividis', vmax=0.9)

        ax_current.set_xticks([])
        ax_current.set_yticks([])
        ax_current.set_xlabel('')
        ax_current.set_ylabel('')

        # Calculate correlation for title
        if title_prefix == "Initial FC":
            corr_value = fc_corr(fc_initial, fc_target)
            title = f"{title_prefix}\nr = {corr_value:.3f}"
        else:
            title = title_prefix

        # Add title as annotation
        ax_current.annotate(title,
                        xy=(0.25, 0.95),
                        xycoords='axes fraction',
                        ha='left', va='top',
                        fontsize=10, fontweight='bold',
                        color='black',
                        bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor='white', alpha=0.9))

    plt.tight_layout()

    # Create grid for parameter exploration
    n = 16

    # Set up parameter axes for exploration
    grid_state = copy.deepcopy(state)
    grid_state.dynamics.w = GridAxis(0.001, 0.7, n)
    grid_state.coupling.instant.G = GridAxis(0.001, 0.7, n)

    # Create space (product creates all combinations of w and G)
    grid = Space(grid_state, mode="product")

    @cache("explore", redo=True)
    def explore():
        # Parallel execution across 8 processes
        exec = ParallelExecution(loss, grid)
        # Alternative: Sequential execution
        # exec = SequentialExecution(loss, grid)
        return exec.run()

    exploration_results = explore()


    # Prepare data for visualization
    pc = grid.collect()
    G_vals = pc.coupling.instant.G.flatten()
    w_vals = pc.dynamics.w.flatten()

    # Get parameter ranges
    G_min, G_max = min(G_vals), max(G_vals)
    w_min, w_max = min(w_vals), max(w_vals)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 4))

    # Create the heatmap
    im = ax.imshow(jnp.stack(exploration_results).reshape(n, n).T,
                cmap='cividis_r',
                extent=[G_min, G_max, w_min, w_max],
                origin='lower',
                aspect='auto',
                interpolation='none')

    # Add colorbar and labels
    cbar = plt.colorbar(im, label="Loss (RMSE)")
    ax.set_xlabel('Global Coupling (G)')
    ax.set_ylabel('Excitatory Recurrence (w)')
    ax.set_title("Parameter Exploration")

    plt.tight_layout()

    # Mark parameters as optimizable
    state.coupling.instant.G = Parameter(state.coupling.instant.G)
    state.dynamics.w = Parameter(state.dynamics.w)
    # Create and run optimizer
    cb = MultiCallback([
        DefaultPrintCallback(every=10),
        SavingCallback(key="state", save_fun=lambda *args: args[1])  # Save updated state
    ])

    @cache("optimize", redo=True)
    def optimize():
        opt = OptaxOptimizer(loss, optax.adam(0.01), callback=cb)
        fitted_state, fitting_data = opt.run(state, max_steps=1000)
        return fitted_state, fitting_data

    fitted_state, fitting_data = optimize()


    # Prepare data for visualization
    pc = grid.collect()
    G_vals = pc.coupling.instant.G
    w_vals = pc.dynamics.w

    # Get parameter ranges
    G_min, G_max = min(G_vals), max(G_vals)
    w_min, w_max = min(w_vals), max(w_vals)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))

    # Create the heatmap
    im = ax.imshow(jnp.stack(exploration_results).reshape(n, n).T,
                cmap='cividis_r',
                extent=[G_min, G_max, w_min, w_max],
                origin='lower',
                aspect='auto',
                interpolation='none')

    # Mark initial value
    G_init = state.coupling.instant.G.value
    w_init = state.dynamics.w.value
    ax.scatter(G_init, w_init, color='white', s=100, marker='o',
            edgecolors='k', linewidths=2, zorder=5)

    # Add annotation
    ax.annotate('Initial', xy=(G_init, w_init),
                xytext=(G_init, w_init+0.05*(w_max-w_min)),
                color='white', fontweight='bold', ha='center', zorder=5,
                path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])

    # Add fitted value point
    G_fit = fitted_state.coupling.instant.G.value
    w_fit = fitted_state.dynamics.w.value
    ax.scatter(G_fit, w_fit, color='white', s=100, marker='o',
            edgecolors='k', linewidths=2, zorder=5)

    # Add annotation for the fitted value
    ax.annotate('Optimized', xy=(G_fit, w_fit),
                xytext=(G_fit, w_fit-0.08*(w_max-w_min)),
                color='white', fontweight='bold', ha='center', zorder=5,
                path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])

    # Add optimization path points
    G_route = np.array([ds.coupling.instant.G.value for ds in fitting_data["state"].save])
    w_route = np.array([ds.dynamics.w.value for ds in fitting_data["state"].save])
    ax.scatter(G_route[::2], w_route[::2], color='white', s=15, marker='o',
            linewidths=1, zorder=4, edgecolors='k')

    # Remove axes ticks and labels
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.tight_layout()

    fitted_state_het = copy.deepcopy(fitted_state)

    # Make w regional (one value per node)
    fitted_state_het.dynamics.w.shape = (82,)

    # Also make I_o regional and mark as optimizable
    fitted_state_het.dynamics.I_o = Parameter(fitted_state_het.dynamics.I_o)
    fitted_state_het.dynamics.I_o.shape = (82,)

    # Keep global coupling fixed at optimized value
    fitted_state_het.coupling.instant.G = fitted_state_het.coupling.instant.G.value

    show_parameters(fitted_state_het)


    @cache("optimize_het", redo=True)
    def optimize_het():
        opt = OptaxOptimizer(loss, optax.adam(0.004, b2=0.999), callback=cb)
        fitted_state, fitting_data = opt.run(fitted_state_het, max_steps=2000)
        return fitted_state, fitting_data

    fitted_state_het, fitting_data_het = optimize_het()

    # Compute FC for both optimization strategies
    fc_global = np.array(observation(fitted_state))
    fc_regional = np.array(observation(fitted_state_het))

    # %%

    # Create the figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.1, 3.54375))

    # Plot the FC matrices
    for ax_current, fc_matrix, title_prefix in zip([ax1, ax2, ax3], [fc_target, fc_global, fc_regional], ["Target FC", "Global Parameters", "Regional Parameters"]):
        fc_matrix = np.copy(fc_matrix)
        np.fill_diagonal(fc_matrix, np.nan)  # Set diagonal to NaN
        im = ax_current.imshow(fc_matrix, cmap='cividis', vmax=1.0)

        ax_current.set_xticks([])
        ax_current.set_yticks([])
        ax_current.set_xlabel('')
        ax_current.set_ylabel('')

        # Calculate correlation for title (if not target)
        if title_prefix == "Target FC":
            title = title_prefix
        elif title_prefix == "Global Parameters":
            corr_value = fc_corr(fc_global, fc_target)
            title = f"{title_prefix}\nr = {corr_value:.3f}"
        else:
            corr_value = fc_corr(fc_regional, fc_target)
            title = f"{title_prefix}\nr = {corr_value:.3f}"

        # Set title
        ax_current.set_title(title, fontsize=10, fontweight='bold')

    plt.tight_layout()


    # Create figure with two scatter plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.1, 5.4), sharey=True, sharex=True)

    # Get upper triangular indices (exclude diagonal)
    triu_idx = np.triu_indices_from(fc_target, k=1)

    # Extract upper triangular values
    fc_target_triu = fc_target[triu_idx]
    fc_global_triu = fc_global[triu_idx]
    fc_regional_triu = fc_regional[triu_idx]

    # Plot global parameters
    ax1.scatter(fc_target_triu, fc_global_triu, alpha=0.3, s=10, color='royalblue', edgecolors='none')
    ax1.plot([fc_target_triu.min(), fc_target_triu.max()],
            [fc_target_triu.min(), fc_target_triu.max()],
            'k--', linewidth=1.5, label='Perfect fit')
    corr_global = fc_corr(fc_global, fc_target)
    ax1.set_xlabel('Empirical FC')
    ax1.set_ylabel('Simulated FC')
    ax1.set_title(f'Global Parameters\nr = {corr_global:.3f}')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # Plot regional parameters
    ax2.scatter(fc_target_triu, fc_regional_triu, alpha=0.3, s=10, color='royalblue', edgecolors='none')
    ax2.plot([fc_target_triu.min(), fc_target_triu.max()],
            [fc_target_triu.min(), fc_target_triu.max()],
            'k--', linewidth=1.5, label='Perfect fit')
    corr_regional = fc_corr(fc_regional, fc_target)
    ax2.set_xlabel('Empirical FC')
    ax2.set_ylabel('Simulated FC')
    ax2.set_title(f'Regional Parameters\nr = {corr_regional:.3f}')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()


    # Calculate mean incoming connectivity for each region
    mean_connectivity = np.mean(conn.weights, axis=1)

    # Extract fitted regional parameters
    w_fitted = fitted_state_het.dynamics.w.value.flatten()
    I_o_fitted = fitted_state_het.dynamics.I_o.value.flatten()

    # Get global optimization values for reference
    w_global = fitted_state.dynamics.w.value
    I_o_global = fitted_state.dynamics.I_o  # Not optimized in global fit, but initial value

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.1, 3.24))

    # Plot w vs mean connectivity
    ax1.scatter(mean_connectivity, w_fitted, alpha=0.7, s=30, color='royalblue', edgecolors='k', linewidths=0.5)
    ax1.axhline(w_global, color='red', linestyle='--', linewidth=2, label=f'Global w = {w_global:.3f}')
    ax1.set_xlabel('Mean Incoming Connectivity')
    ax1.set_ylabel('Fitted w (Excitatory Recurrence)')
    ax1.set_title('Regional Excitatory Recurrence Parameters')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot I_o vs mean connectivity
    ax2.scatter(mean_connectivity, I_o_fitted, alpha=0.7, s=30, color='royalblue', edgecolors='k', linewidths=0.5)
    ax2.axhline(I_o_global, color='red', linestyle='--', linewidth=2, label=f'Initial I_o = {I_o_global:.3f}')
    ax2.set_xlabel('Mean Incoming Connectivity')
    ax2.set_ylabel('Fitted I_o (External Input)')
    ax2.set_title('Regional External Input Parameters')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()


