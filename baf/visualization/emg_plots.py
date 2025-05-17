import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class EMGPlotter:
    """
    Electromyography (EMG) visualization tool.
    
    This class provides methods to create standardized visualizations of 
    muscle activation data during gait or other movements.
    """
    
    def __init__(self, figsize=(10, 6)):
        """
        Initialize the EMGPlotter with default figure size.
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height) in inches
        """
        self.figsize = figsize
        self.fig = None
        self.axes = None
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    
    def plot_emg(self, data, 
                 muscle_cols=None,
                 gait_events=None,
                 title='Muscle Activity During Gait Cycle',
                 normalize=True,
                 show_legend=True):
        """
        Plot EMG data for multiple muscles.
        
        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing EMG data
        muscle_cols : dict or list
            If dict: {muscle_name: column_name}
            If list: list of column names (will use column names as labels)
            If None: use all columns in data
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
        
        # Determine which columns to plot
        if muscle_cols is None:
            # Use all columns
            muscle_cols = {col: col for col in data.columns}
        elif isinstance(muscle_cols, list):
            # Convert list to dict
            muscle_cols = {col: col for col in muscle_cols}
        
        # Plot each muscle's EMG
        for i, (muscle, col) in enumerate(muscle_cols.items()):
            color = self.colors[i % len(self.colors)]
            self.axes.plot(data.index, data[col], f'{color}-', label=muscle.replace('_', ' ').title())
        
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
        
        self.axes.set_ylabel('Normalized EMG')
        self.axes.set_title(title)
        self.axes.grid(True)
        
        # Set y-axis limits for normalized EMG
        if normalize:
            self.axes.set_ylim([-0.05, 1.05])
        
        if show_legend:
            self.axes.legend()
        
        plt.tight_layout()
        
        return self.fig, self.axes
    
    def plot_emg_grid(self, data, muscle_cols=None, gait_events=None, ncols=2, 
                     title='Muscle Activity', normalize=True):
        """
        Plot multiple muscles in a grid layout.
        
        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing EMG data
        muscle_cols : dict or list
            If dict: {muscle_name: column_name}
            If list: list of column names (will use column names as labels)
            If None: use all columns in data
        gait_events : dict
            Dictionary of gait events with event name as key and percent of gait cycle as value
        ncols : int
            Number of columns in the grid
        title : str
            Main title for the figure
        normalize : bool
            If True, assumes data is normalized to gait cycle (0-100%)
            
        Returns
        -------
        tuple
            (fig, axes) matplotlib figure and axes objects
        """
        # Determine which columns to plot
        if muscle_cols is None:
            # Use all columns
            muscle_cols = {col: col for col in data.columns}
        elif isinstance(muscle_cols, list):
            # Convert list to dict
            muscle_cols = {col: col for col in muscle_cols}
        
        # Calculate grid dimensions
        n_muscles = len(muscle_cols)
        nrows = (n_muscles + ncols - 1) // ncols
        
        # Create figure with subplots
        figsize = (self.figsize[0], self.figsize[1] * nrows / 2)
        self.fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
        
        # Flatten axes array for easy indexing
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each muscle
        for i, (muscle, col) in enumerate(muscle_cols.items()):
            if i < len(axes):
                ax = axes[i]
                ax.plot(data.index, data[col], 'b-')
                
                # Add gait events if provided
                if gait_events:
                    for event, position in gait_events.items():
                        ax.axvline(x=position, color='k', linestyle='--', alpha=0.7)
                
                # Set labels
                if i >= len(axes) - ncols:  # Only for bottom row
                    if normalize:
                        ax.set_xlabel('Gait Cycle (%)')
                    else:
                        ax.set_xlabel('Time (s)')
                
                ax.set_ylabel('EMG')
                ax.set_title(muscle.replace('_', ' ').title())
                ax.grid(True)
                
                # Set y-axis limits for normalized EMG
                if normalize:
                    ax.set_ylim([-0.05, 1.05])
        
        # Hide unused subplots
        for j in range(n_muscles, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for the suptitle
        
        return self.fig, axes
    
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