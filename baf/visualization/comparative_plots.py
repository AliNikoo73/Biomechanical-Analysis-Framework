import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class ComparativePlotter:
    """
    Comparative visualization tool for biomechanical data.
    
    This class provides methods to create combined visualizations that integrate
    multiple data types (kinematics, kinetics, EMG) for comprehensive analysis.
    """
    
    def __init__(self, figsize=(12, 10)):
        """
        Initialize the ComparativePlotter with default figure size.
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height) in inches
        """
        self.figsize = figsize
        self.fig = None
        self.axes = None
    
    def plot_gait_analysis(self, data, 
                          joint_cols=None,
                          grf_cols=None,
                          emg_cols=None,
                          gait_events=None,
                          title='Gait Analysis',
                          normalize=True):
        """
        Create a comprehensive gait analysis visualization with joint angles, GRF, and EMG.
        
        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing all data
        joint_cols : dict
            Dictionary mapping joint names to column names, e.g., {'hip': 'hip_angle'}
        grf_cols : dict
            Dictionary mapping GRF components to column names
        emg_cols : dict
            Dictionary mapping muscle names to column names
        gait_events : dict
            Dictionary of gait events with event name as key and percent of gait cycle as value
        title : str
            Main title for the figure
        normalize : bool
            If True, assumes data is normalized to gait cycle (0-100%)
            
        Returns
        -------
        tuple
            (fig, axes) matplotlib figure and axes objects
        """
        # Determine number of subplots needed
        n_plots = 0
        if joint_cols:
            n_plots += 1
        if grf_cols:
            n_plots += 1
        if emg_cols:
            n_plots += 1
            
        if n_plots == 0:
            raise ValueError("At least one of joint_cols, grf_cols, or emg_cols must be provided")
        
        # Create figure with subplots
        self.fig, self.axes = plt.subplots(n_plots, 1, figsize=self.figsize, sharex=True)
        
        # Ensure axes is always a list/array
        if n_plots == 1:
            self.axes = np.array([self.axes])
        
        # Plot counter
        plot_idx = 0
        
        # Joint angles
        if joint_cols:
            ax = self.axes[plot_idx]
            for joint, col in joint_cols.items():
                if joint.lower() == 'hip':
                    color = 'r'
                elif joint.lower() == 'knee':
                    color = 'g'
                elif joint.lower() == 'ankle':
                    color = 'b'
                else:
                    color = 'k'
                ax.plot(data.index, data[col], f'{color}-', label=joint.title())
            
            # Add gait events if provided
            if gait_events:
                for event, position in gait_events.items():
                    ax.axvline(x=position, color='k', linestyle='--', alpha=0.7)
            
            ax.set_ylabel('Angle (deg)')
            ax.set_title('Joint Angles')
            ax.grid(True)
            ax.legend()
            plot_idx += 1
        
        # Ground reaction forces
        if grf_cols:
            ax = self.axes[plot_idx]
            for force, col in grf_cols.items():
                if 'vertical' in force.lower():
                    color = 'r'
                elif 'anterior' in force.lower() or 'posterior' in force.lower():
                    color = 'g'
                elif 'medial' in force.lower() or 'lateral' in force.lower():
                    color = 'b'
                else:
                    color = 'k'
                
                # Shorten label if too long
                label = force.replace('_', ' ').title()
                if len(label) > 15:
                    if 'anterior' in force.lower() or 'posterior' in force.lower():
                        label = 'A-P'
                    elif 'medial' in force.lower() or 'lateral' in force.lower():
                        label = 'M-L'
                
                ax.plot(data.index, data[col], f'{color}-', label=label)
            
            # Add gait events if provided
            if gait_events:
                for event, position in gait_events.items():
                    ax.axvline(x=position, color='k', linestyle='--', alpha=0.7)
            
            ax.set_ylabel('Force (N)')
            ax.set_title('Ground Reaction Forces')
            ax.grid(True)
            ax.legend()
            plot_idx += 1
        
        # EMG
        if emg_cols:
            ax = self.axes[plot_idx]
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink']
            
            for i, (muscle, col) in enumerate(emg_cols.items()):
                color = colors[i % len(colors)]
                
                # Shorten label if too long
                label = muscle.replace('_', ' ').title()
                if len(label) > 15:
                    if 'gastrocnemius' in muscle.lower():
                        label = 'Gastroc'
                    elif 'tibialis' in muscle.lower() and 'anterior' in muscle.lower():
                        label = 'Tib Ant'
                    elif 'rectus' in muscle.lower() and 'femoris' in muscle.lower():
                        label = 'Rect Fem'
                    elif 'vastus' in muscle.lower() and 'lateralis' in muscle.lower():
                        label = 'Vast Lat'
                
                ax.plot(data.index, data[col], f'{color}-', label=label)
            
            # Add gait events if provided
            if gait_events:
                for event, position in gait_events.items():
                    ax.axvline(x=position, color='k', linestyle='--', alpha=0.7)
            
            ax.set_ylabel('Normalized EMG')
            ax.set_title('Muscle Activity')
            ax.grid(True)
            ax.legend()
            
            # Set y-axis limits for normalized EMG
            if normalize:
                ax.set_ylim([-0.05, 1.05])
        
        # Set x-axis label on bottom subplot
        if normalize:
            self.axes[-1].set_xlabel('Gait Cycle (%)')
        else:
            self.axes[-1].set_xlabel('Time (s)')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Make room for the suptitle
        
        return self.fig, self.axes
    
    def plot_condition_comparison(self, data_dict, plot_type, 
                                 column, 
                                 gait_events=None,
                                 title=None,
                                 normalize=True,
                                 show_legend=True):
        """
        Compare the same variable across different conditions.
        
        Parameters
        ----------
        data_dict : dict
            Dictionary mapping condition names to DataFrames
        plot_type : str
            Type of plot: 'joint_angle', 'grf', or 'emg'
        column : str or dict
            If str: Column name to plot for all conditions
            If dict: Dictionary mapping condition names to column names
        gait_events : dict
            Dictionary of gait events with event name as key and percent of gait cycle as value
        title : str
            Plot title
        normalize : bool
            If True, assumes data is normalized to gait cycle (0-100%)
        show_legend : bool
            If True, show legend
            
        Returns
        -------
        tuple
            (fig, ax) matplotlib figure and axes objects
        """
        self.fig, self.axes = plt.subplots(figsize=self.figsize)
        
        # Set default title based on plot type and column
        if title is None:
            if isinstance(column, str):
                col_name = column
            else:
                # Use the first column name for the title
                col_name = next(iter(column.values())) if column else "Data"
                
            if plot_type == 'joint_angle':
                title = f"{col_name.replace('_', ' ').title()} Comparison"
            elif plot_type == 'grf':
                title = f"{col_name.replace('_', ' ').title()} Comparison"
            elif plot_type == 'emg':
                title = f"{col_name.replace('_', ' ').title()} Activity Comparison"
        
        # Plot each condition
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink']
        for i, (condition, data) in enumerate(data_dict.items()):
            color = colors[i % len(colors)]
            
            # Determine which column to use for this condition
            if isinstance(column, dict):
                if condition in column:
                    col = column[condition]
                else:
                    # Skip this condition if no column is specified
                    continue
            else:
                # Use the same column for all conditions
                col = column
                
            # Check if the column exists in the dataframe
            if col in data.columns:
                self.axes.plot(data.index, data[col], f'{color}-', label=condition)
            else:
                print(f"Warning: Column '{col}' not found in data for condition '{condition}'")
        
        # Add gait events if provided
        if gait_events:
            for event, position in gait_events.items():
                self.axes.axvline(x=position, color='k', linestyle='--', alpha=0.7,
                                 label=event.replace('_', ' ').title())
        
        # Set labels and title
        if normalize:
            self.axes.set_xlabel('Gait Cycle (%)')
        else:
            self.axes.set_xlabel('Time (s)')
        
        # Set y-axis label based on plot type
        if plot_type == 'joint_angle':
            self.axes.set_ylabel('Angle (deg)')
        elif plot_type == 'grf':
            self.axes.set_ylabel('Force (N)')
        elif plot_type == 'emg':
            self.axes.set_ylabel('Normalized EMG')
            # Set y-axis limits for normalized EMG
            if normalize:
                self.axes.set_ylim([-0.05, 1.05])
        
        self.axes.set_title(title)
        self.axes.grid(True)
        
        if show_legend:
            self.axes.legend()
        
        plt.tight_layout()
        
        return self.fig, self.axes
    
    def save_figure(self, filename, dpi=300):
        """
        Save the figure to a file.
        
        Parameters
        ----------
        filename : str
            Path to save the figure
        dpi : int
            Resolution in dots per inch
        """
        if self.fig is None:
            raise ValueError("No figure to save. Create a plot first.")
        
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {filename}") 