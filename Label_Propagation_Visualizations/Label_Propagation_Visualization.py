"""
Label Propagation State Visualization Script

This script visualizes specific shots from the LABEL_PROPAGATED_DATABASE.csv file,
plotting fs04 values with state classifications shown as color overlays.

State Classifications:
- State -1: Unknown/Unlabeled (Gray)
- State 1: Suppressed (Green)
- State 2: Dithering (Gold)
- State 3: Mitigated (Tomato)
- State 4: ELMing (Crimson)

HOW TO USE:
1. Run the script: python Label_Propagation_Visualization.py
2. The script will visualize the specified shots and create a visualization
3. Output is saved to the 'Label_Propagation_Visualizations/' folder
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
plt.close('all')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Specific shot numbers to visualize
SHOT_NUMBERS = [195955, 195959, 196037, 196038, 198851, 199210]

# Output directory
OUTPUT_DIR = 'Label_Propagation_Visualizations'

# =============================================================================


def load_label_propagated_data():
    """Load the label propagated database"""
    print("Loading Label Propagated Database...")
    
    # Try multiple possible paths for the data file
    possible_paths = [
        'LABEL_PROPAGATED_DATABASE.csv',     # Current directory
        '/mnt/homes/sr4240/my_folder/LABEL_PROPAGATED_DATABASE.csv'  # Absolute path
    ]
    
    df = None
    for path in possible_paths:
        try:
            print(f"Trying to load from: {path}")
            df = pd.read_csv(path)
            print(f"Successfully loaded data from: {path}")
            print(f"Loaded {len(df)} rows")
            break
        except FileNotFoundError:
            print(f"File not found at: {path}")
            continue
    
    if df is None:
        raise FileNotFoundError("Could not find LABEL_PROPAGATED_DATABASE.csv in any of the expected locations")
    
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df


def validate_shots(df, shot_numbers):
    """Validate that the specified shots exist in the dataset"""
    unique_shots = df['shot'].unique()
    print(f"Total unique shots in dataset: {len(unique_shots)}")
    
    # Check which shots exist in the dataset
    valid_shots = [shot for shot in shot_numbers if shot in unique_shots]
    missing_shots = [shot for shot in shot_numbers if shot not in unique_shots]
    
    if missing_shots:
        print(f"Warning: The following shots were not found in the dataset: {missing_shots}")
    
    print(f"Shots to visualize: {valid_shots}")
    
    return valid_shots


def extract_shot_data(df, shot_number):
    """Extract all time series data for a specific shot"""
    # Filter data for the specific shot
    shot_data = df[df['shot'] == shot_number].copy()
    
    # Sort by time to ensure proper time series order
    shot_data = shot_data.sort_values('time').reset_index(drop=True)
    
    return shot_data


def create_multi_shot_visualization(df, selected_shots):
    """Create a visualization of multiple shots with fs04 and state color overlays"""
    print("Creating multi-shot visualization...")
    
    num_shots = len(selected_shots)
    
    # Create figure with subplots - one row per shot
    fig, axes = plt.subplots(num_shots, 1, figsize=(24, 5 * num_shots))
    fig.suptitle('Label Propagated Database - Shot State Classifications', 
                 fontsize=22, fontweight='bold', y=0.99)
    
    # Handle case where only one shot (axes is not a list)
    if num_shots == 1:
        axes = [axes]
    
    # Define colors for states (including -1 for unknown/unlabeled)
    state_colors = {
        -1: '#808080',  # Gray for unknown/unlabeled
        1: '#2E8B57',   # Green for Suppressed
        2: '#FFD700',   # Gold for Dithering
        3: '#FF6347',   # Tomato for Mitigated
        4: '#DC143C'    # Crimson for ELMing
    }
    state_names = {
        -1: 'Unknown',
        1: 'Suppressed',
        2: 'Dithering',
        3: 'Mitigated',
        4: 'ELMing'
    }
    
    # Plot each shot
    for idx, (ax, shot_number) in enumerate(zip(axes, selected_shots)):
        print(f"Processing shot {shot_number} ({idx + 1}/{num_shots})...")
        
        # Extract data for this shot
        shot_data = extract_shot_data(df, shot_number)
        
        if len(shot_data) == 0:
            ax.text(0.5, 0.5, f'No data for shot {shot_number}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Shot {shot_number} - No Data', fontsize=16, fontweight='bold')
            continue
        
        time = shot_data['time']
        fs04 = shot_data['fs04']
        states = shot_data['state']
        
        # Get y-axis limits for fill_between
        fs04_min = fs04.min()
        fs04_max = fs04.max()
        
        # Add some padding to y limits
        y_padding = (fs04_max - fs04_min) * 0.05
        y_min = fs04_min - y_padding
        y_max = fs04_max + y_padding
        
        # Plot fs04 time series
        ax.plot(time, fs04, 'k-', linewidth=1.5, alpha=0.8, label='fs04', zorder=2)
        
        # Color background by state
        for state in [-1, 1, 2, 3, 4]:
            state_mask = states == state
            if state_mask.any():
                ax.fill_between(time, y_min, y_max, 
                               where=state_mask, alpha=0.35, 
                               color=state_colors[state], 
                               label=f'{state_names[state]}',
                               zorder=1)
        
        # Calculate state distribution for this shot
        state_counts = states.value_counts().to_dict()
        state_info = ', '.join([f'{state_names.get(k, k)}: {v}' for k, v in sorted(state_counts.items())])
        
        ax.set_ylabel('fs04', fontsize=14)
        ax.set_title(f'Shot {shot_number} - Time Points: {len(shot_data)} | States: {state_info}', 
                    fontsize=16, fontweight='bold')
        
        # Position legend outside the plot area
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11, 
                 frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_min, y_max)
        
        # Add x-axis label only for the last plot
        if idx == num_shots - 1:
            ax.set_xlabel('Time (ms)', fontsize=14)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(right=0.88, hspace=0.35, bottom=0.05, top=0.95)
    
    return fig


def print_shot_statistics(df, selected_shots):
    """Print statistics about the selected shots"""
    print("\n" + "=" * 70)
    print("SELECTED SHOTS STATISTICS")
    print("=" * 70)
    
    state_names = {
        -1: 'Unknown',
        1: 'Suppressed',
        2: 'Dithering',
        3: 'Mitigated',
        4: 'ELMing'
    }
    
    for shot_number in selected_shots:
        shot_data = df[df['shot'] == shot_number]
        print(f"\nShot {shot_number}:")
        print(f"  Time points: {len(shot_data)}")
        print(f"  Time range: {shot_data['time'].min():.1f} to {shot_data['time'].max():.1f} ms")
        print(f"  State distribution:")
        for state, count in sorted(shot_data['state'].value_counts().items()):
            state_name = state_names.get(state, f'State {state}')
            percentage = 100 * count / len(shot_data)
            print(f"    {state_name}: {count} ({percentage:.1f}%)")


def main():
    """Main function to run the visualization"""
    print("=" * 70)
    print("Label Propagation State Visualization")
    print("=" * 70)
    
    # Load data
    df = load_label_propagated_data()
    
    # Validate and get specified shots
    selected_shots = validate_shots(df, SHOT_NUMBERS)
    
    # Print statistics
    print_shot_statistics(df, selected_shots)
    
    # Create visualization
    fig = create_multi_shot_visualization(df, selected_shots)
    
    # Create output directory if it doesn't exist
    output_dir = OUTPUT_DIR
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\nCreated directory: {output_dir}")
    
    # Save the figure
    output_filename = os.path.join(output_dir, 'specified_shots_state_visualization.png')
    fig.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved to: {output_filename}")
    
    # Close the figure to free memory
    plt.close(fig)
    
    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)
    
    return selected_shots


if __name__ == "__main__":
    selected_shots = main()

