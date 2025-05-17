import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class GRFPlotter:
    """
    Ground Reaction Force visualization tool.
    
    This class provides methods to create standardized visualizations of 
    ground reaction force data during gait or other movements.
    """
    
    def __init__(self, figsize=(10, 6)):
        """
        Initialize the GRFPlotter with default figure size.
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height) in inches
        """
        self.figsize = figsize
        self.fig = None
        self.axes = None
    
    def plot_grf(self, data, 
                 vertical_col='vertical_force', 
                 ap_col='anterior_posterior_force', 
                 ml_col='medial_lateral_force',
                 gait_events=None,
                 title='Ground Reaction Forces During Gait Cycle',
                 normalize=True,
                 show_legend=True):
        """
        Plot ground reaction forces.
        
        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing GRF data
        vertical_col : str
            Column name for vertical force
        ap_col : str
            Column name for anterior-posterior force
        ml_col : str
            Column name for medial-lateral force
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
        
        # Plot forces
        self.axes.plot(data.index, data[vertical_col], 'r-', label='Vertical')
        self.axes.plot(data.index, data[ap_col], 'g-', label='Anterior-Posterior')
        self.axes.plot(data.index, data[ml_col], 'b-', label='Medial-Lateral')
        
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
        
        self.axes.set_ylabel('Force (N)')
        self.axes.set_title(title)
        self.axes.grid(True)
        
        if show_legend:
            self.axes.legend()
        
        plt.tight_layout()
        
        return self.fig, self.axes
    
    def add_bodyweight_normalization(self, bodyweight):
        """
        Add a secondary y-axis showing force normalized to bodyweight.
        
        Parameters
        ----------
        bodyweight : float
            Subject's bodyweight in N
            
        Returns
        -------
        matplotlib.axes.Axes
            Secondary y-axis
        """
        if self.axes is None:
            raise ValueError("Plot must be created before adding bodyweight normalization")
        
        # Create secondary y-axis
        ax2 = self.axes.twinx()
        
        # Get current y limits and convert to bodyweight
        y1_min, y1_max = self.axes.get_ylim()
        ax2.set_ylim(y1_min / bodyweight, y1_max / bodyweight)
        
        ax2.set_ylabel('Force (× BW)')
        
        return ax2
    
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