"""
Random Shot Overlayed Visualization Script

This script randomly selects 10 shots out of ~2000 total shots and creates 
one visualization per shot showing fs04, density, betan, and Ip time series 
overlayed with the state classification.

Each image shows 4 subplots for a single shot, with the background colored
by the plasma state classification.

HOW TO USE:
1. Run the script: python Random_Shot_Overlayed_Visualization.py
2. Images will be saved to the 'Random_Shot_Plots' directory
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_RANDOM_SHOTS = 10  # Number of random shots to visualize
OUTPUT_DIR = 'Random_Shot_Plots'

# State colors and names
STATE_COLORS = {
    1: '#2E8B57',   # Sea Green - Suppressed
    2: '#FFD700',   # Gold - Dithering  
    3: '#FF6347',   # Tomato - Mitigated
    4: '#DC143C'    # Crimson - ELMing
}
STATE_NAMES = {
    1: 'Suppressed',
    2: 'Dithering', 
    3: 'Mitigated',
    4: 'ELMing'
}


def load_data():
    """Load the plasma data from CSV"""
    print("Loading plasma data...")
    
    possible_paths = [
        'plasma_data.csv',
        '/mnt/homes/sr4240/my_folder/plasma_data.csv',
        '../plasma_data.csv'
    ]
    
    df = None
    for path in possible_paths:
        try:
            df = pd.read_csv(path)
            print(f"Successfully loaded data from: {path}")
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        raise FileNotFoundError("Could not find plasma_data.csv")
    
    # Remove problematic shot
    df = df[df['shot'] != 191675].copy()
    
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Filter to n=3 and non-zero states
    df = df[df['n'] == 3]
    df = df[df['state'] != 0]
    
    print(f"Loaded {len(df)} rows")
    print(f"Total unique shots: {df['shot'].nunique()}")
    
    return df


def select_random_shots(df, n_shots=N_RANDOM_SHOTS):
    """Randomly select n_shots from all available shots"""
    unique_shots = df['shot'].unique()
    total_shots = len(unique_shots)
    
    print(f"\nSelecting {n_shots} random shots from {total_shots} total shots...")
    
    # Randomly select shots
    selected_shots = np.random.choice(unique_shots, size=min(n_shots, total_shots), replace=False)
    selected_shots = sorted(selected_shots)
    
    print(f"Selected shots: {selected_shots}")
    
    return selected_shots


def create_shot_visualization(df, shot_number, output_dir):
    """
    Create a 4-subplot figure for a single shot showing:
    - fs04
    - density  
    - betan
    - Ip
    All overlayed with state classification coloring.
    """
    # Extract data for this shot
    shot_data = df[df['shot'] == shot_number].copy()
    shot_data = shot_data.sort_values('time').reset_index(drop=True)
    
    if len(shot_data) == 0:
        print(f"  No data found for shot {shot_number}, skipping...")
        return None
    
    print(f"  Processing shot {shot_number}: {len(shot_data)} time points")
    
    # Get time and state data
    time = shot_data['time'].values
    states = shot_data['state'].values
    
    # Get the 4 parameters
    fs04 = shot_data['fs04'].values
    density = shot_data['density'].values
    betan = shot_data['betan'].values
    ip = shot_data['Ip'].values  # Capital I
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(f'Shot {shot_number} - Plasma Parameters with State Classification', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Parameters to plot
    params = [
        ('fs04', fs04, 'FS04 Signal'),
        ('density', density, 'Density'),
        ('betan', betan, 'Beta-N'),
        ('Ip', ip, 'Plasma Current (Ip)')
    ]
    
    for ax_idx, (param_name, param_data, param_label) in enumerate(params):
        ax = axes[ax_idx]
        
        # Skip if all NaN
        if np.all(np.isnan(param_data)):
            ax.text(0.5, 0.5, f'No data available for {param_label}', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_ylabel(param_label, fontsize=12)
            continue
        
        # Plot the parameter time series
        ax.plot(time, param_data, 'k-', linewidth=1.2, alpha=0.8, zorder=3)
        
        # Get y-axis limits for fill_between
        valid_data = param_data[~np.isnan(param_data)]
        if len(valid_data) > 0:
            y_min, y_max = np.nanmin(param_data), np.nanmax(param_data)
            y_range = y_max - y_min
            if y_range == 0:
                y_range = abs(y_max) * 0.1 if y_max != 0 else 1
            y_min -= y_range * 0.05
            y_max += y_range * 0.05
        else:
            y_min, y_max = 0, 1
        
        # Color background by state
        for state in [1, 2, 3, 4]:
            state_mask = states == state
            if np.any(state_mask):
                ax.fill_between(time, y_min, y_max,
                              where=state_mask, alpha=0.35,
                              color=STATE_COLORS[state],
                              label=f'{STATE_NAMES[state]} (State {state})',
                              zorder=1)
        
        # Set labels and formatting
        ax.set_ylabel(param_label, fontsize=12, fontweight='bold')
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.tick_params(labelsize=10)
        
        # Add legend only to first subplot
        if ax_idx == 0:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10,
                     frameon=True, fancybox=True, shadow=True)
    
    # Set x-axis label on bottom subplot
    axes[-1].set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.85, hspace=0.08, top=0.94)
    
    # Save figure
    filename = os.path.join(output_dir, f'shot_{shot_number}_overlayed.png')
    fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  Saved: {filename}")
    
    return filename


def main():
    """Main function to run the visualization"""
    print("="*70)
    print("Random Shot Overlayed Visualization")
    print("="*70)
    print(f"Will select {N_RANDOM_SHOTS} random shots and create visualizations")
    print("Parameters: fs04, density, betan, Ip")
    print("="*70 + "\n")
    
    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}\n")
    
    # Load data
    df = load_data()
    
    # Select random shots
    selected_shots = select_random_shots(df, N_RANDOM_SHOTS)
    
    # Create visualizations
    print("\nCreating visualizations...")
    print("-"*50)
    
    successful = 0
    failed = 0
    
    for i, shot_number in enumerate(selected_shots, 1):
        print(f"\n[{i}/{len(selected_shots)}] Shot {shot_number}")
        
        try:
            result = create_shot_visualization(df, shot_number, OUTPUT_DIR)
            if result is not None:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"Total shots processed: {len(selected_shots)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nImages saved to: {OUTPUT_DIR}/")
    print("="*70)


if __name__ == "__main__":
    main()