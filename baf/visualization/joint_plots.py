"""
Joint Angle Visualization

This module provides tools for visualizing joint angles and other kinematic data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, List, Tuple, Union, Optional
import pandas as pd


class JointPlotter:
    """
    Class for creating and customizing joint angle plots.
    """
    
    def __init__(
        self,
        figsize: Tuple[float, float] = (10, 6),
        dpi: int = 100,
        style: str = "seaborn-v0_8-whitegrid",
    ):
        """
        Initialize the JointPlotter.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size (width, height) in inches. Default is (10, 6).
        dpi : int, optional
            Figure resolution. Default is 100.
        style : str, optional
            Matplotlib style to use. Default is "seaborn-v0_8-whitegrid".
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        self.fig = None
        self.axes = None
    
    def create_figure(
        self,
        num_rows: int = 1,
        num_cols: int = 3,
        sharex: bool = True,
        sharey: bool = False,
    ) -> Tuple[Figure, List[Axes]]:
        """
        Create a figure with subplots for joint angle visualization.

        Parameters
        ----------
        num_rows : int, optional
            Number of subplot rows. Default is 1.
        num_cols : int, optional
            Number of subplot columns. Default is 3 (for hip, knee, ankle).
        sharex : bool, optional
            Whether to share x-axis among subplots. Default is True.
        sharey : bool, optional
            Whether to share y-axis among subplots. Default is False.

        Returns
        -------
        fig : Figure
            Matplotlib figure object.
        axes : list of Axes
            List of subplot axes.
        """
        with plt.style.context(self.style):
            self.fig, self.axes = plt.subplots(
                num_rows, num_cols, figsize=self.figsize, dpi=self.dpi,
                sharex=sharex, sharey=sharey
            )
            
            # Convert to list if only one row
            if num_rows == 1 and num_cols == 1:
                self.axes = [self.axes]
            elif num_rows == 1 or num_cols == 1:
                self.axes = self.axes.flatten()
        
        return self.fig, self.axes
    
    def plot_joint_angles(
        self,
        data: pd.DataFrame,
        joint_cols: Dict[str, str] = None,
        gait_events: Dict[str, Union[int, float]] = None,
        labels: Dict[str, str] = None,
        colors: Dict[str, str] = None,
        title: str = "Joint Angles",
        x_label: str = "Gait Cycle (%)",
        y_label: str = "Angle (deg)",
        legend: bool = True,
        grid: bool = True,
        show_events: bool = True,
    ) -> Tuple[Figure, List[Axes]]:
        """
        Plot joint angles for hip, knee, and ankle.

        Parameters
        ----------
        data : DataFrame
            Data containing joint angle columns.
        joint_cols : dict, optional
            Dictionary mapping joint names to column names in data.
            Default is {"hip": "hip_angle", "knee": "knee_angle", "ankle": "ankle_angle"}.
        gait_events : dict, optional
            Dictionary containing gait events as percentages of the gait cycle.
            Example: {"toe_off": 60}.
        labels : dict, optional
            Dictionary mapping joint names to labels for the legend.
            Default is {"hip": "Hip", "knee": "Knee", "ankle": "Ankle"}.
        colors : dict, optional
            Dictionary mapping joint names to colors.
            Default is {"hip": "r", "knee": "g", "ankle": "b"}.
        title : str, optional
            Figure title. Default is "Joint Angles".
        x_label : str, optional
            X-axis label. Default is "Gait Cycle (%)".
        y_label : str, optional
            Y-axis label. Default is "Angle (deg)".
        legend : bool, optional
            Whether to show the legend. Default is True.
        grid : bool, optional
            Whether to show the grid. Default is True.
        show_events : bool, optional
            Whether to show vertical lines for gait events. Default is True.

        Returns
        -------
        fig : Figure
            Matplotlib figure object.
        axes : list of Axes
            List of subplot axes.
        """
        # Set default values if not provided
        if joint_cols is None:
            joint_cols = {"hip": "hip_angle", "knee": "knee_angle", "ankle": "ankle_angle"}
        
        if labels is None:
            labels = {"hip": "Hip Flexion/Extension", "knee": "Knee Flexion/Extension", "ankle": "Ankle Dorsi/Plantarflexion"}
        
        if colors is None:
            colors = {"hip": "r", "knee": "g", "ankle": "b"}
        
        # Create figure if not already created
        if self.fig is None or self.axes is None:
            self.create_figure(num_cols=len(joint_cols))
        
        # Plot each joint angle
        for i, (joint, col) in enumerate(joint_cols.items()):
            if col in data.columns:
                self.axes[i].plot(data.index, data[col], color=colors.get(joint, "k"), label=labels.get(joint, joint))
                self.axes[i].set_title(labels.get(joint, joint))
                
                # Set axis labels
                self.axes[i].set_xlabel(x_label)
                self.axes[i].set_ylabel(y_label)
                
                # Show gait events if provided
                if show_events and gait_events is not None:
                    for event, value in gait_events.items():
                        self.axes[i].axvline(x=value, color="k", linestyle="--", alpha=0.7, label=event)
                
                # Set grid
                self.axes[i].grid(grid)
                
                # Show legend
                if legend:
                    self.axes[i].legend()
        
        # Set figure title
        self.fig.suptitle(title, fontsize=14)
        self.fig.tight_layout()
        
        return self.fig, self.axes
    
    def plot_comparison(
        self,
        data_list: List[pd.DataFrame],
        labels: List[str],
        joint_cols: Dict[str, str] = None,
        colors: List[str] = None,
        styles: List[str] = None,
        title: str = "Joint Angle Comparison",
        x_label: str = "Gait Cycle (%)",
        y_label: str = "Angle (deg)",
        legend: bool = True,
        grid: bool = True,
    ) -> Tuple[Figure, List[Axes]]:
        """
        Plot a comparison of joint angles from multiple datasets.

        Parameters
        ----------
        data_list : list of DataFrames
            List of DataFrames containing joint angle data.
        labels : list of str
            Labels for each dataset in the legend.
        joint_cols : dict, optional
            Dictionary mapping joint names to column names in data.
            Default is {"hip": "hip_angle", "knee": "knee_angle", "ankle": "ankle_angle"}.
        colors : list of str, optional
            List of colors for each dataset. Default is ["r", "b", "g", "c", "m"].
        styles : list of str, optional
            List of line styles for each dataset. Default is ["-", "--", ":", "-."].
        title : str, optional
            Figure title. Default is "Joint Angle Comparison".
        x_label : str, optional
            X-axis label. Default is "Gait Cycle (%)".
        y_label : str, optional
            Y-axis label. Default is "Angle (deg)".
        legend : bool, optional
            Whether to show the legend. Default is True.
        grid : bool, optional
            Whether to show the grid. Default is True.

        Returns
        -------
        fig : Figure
            Matplotlib figure object.
        axes : list of Axes
            List of subplot axes.
        """
        # Set default values if not provided
        if joint_cols is None:
            joint_cols = {"hip": "hip_angle", "knee": "knee_angle", "ankle": "ankle_angle"}
        
        if colors is None:
            colors = ["r", "b", "g", "c", "m", "y", "k"]
        
        if styles is None:
            styles = ["-", "--", ":", "-."]
        
        # Create figure if not already created
        if self.fig is None or self.axes is None:
            self.create_figure(num_cols=len(joint_cols))
        
        # Plot each joint angle for each dataset
        for i, (joint, col) in enumerate(joint_cols.items()):
            for j, (data, label) in enumerate(zip(data_list, labels)):
                if col in data.columns:
                    color = colors[j % len(colors)]
                    style = styles[j % len(styles)]
                    self.axes[i].plot(data.index, data[col], color=color, linestyle=style, label=f"{label}")
            
            self.axes[i].set_title(joint.capitalize())
            self.axes[i].set_xlabel(x_label)
            self.axes[i].set_ylabel(y_label)
            self.axes[i].grid(grid)
            
            if legend:
                self.axes[i].legend()
        
        # Set figure title
        self.fig.suptitle(title, fontsize=14)
        self.fig.tight_layout()
        
        return self.fig, self.axes
    
    def add_normal_range(
        self,
        mean_data: pd.DataFrame,
        std_data: pd.DataFrame,
        joint_cols: Dict[str, str] = None,
        colors: Dict[str, str] = None,
        alpha: float = 0.2,
    ) -> None:
        """
        Add normal range (mean ± std) to the joint angle plots.

        Parameters
        ----------
        mean_data : DataFrame
            DataFrame containing mean joint angle data.
        std_data : DataFrame
            DataFrame containing standard deviation of joint angle data.
        joint_cols : dict, optional
            Dictionary mapping joint names to column names in data.
            Default is {"hip": "hip_angle", "knee": "knee_angle", "ankle": "ankle_angle"}.
        colors : dict, optional
            Dictionary mapping joint names to colors.
            Default is {"hip": "r", "knee": "g", "ankle": "b"}.
        alpha : float, optional
            Alpha value for the fill between. Default is 0.2.
        """
        # Set default values if not provided
        if joint_cols is None:
            joint_cols = {"hip": "hip_angle", "knee": "knee_angle", "ankle": "ankle_angle"}
        
        if colors is None:
            colors = {"hip": "r", "knee": "g", "ankle": "b"}
        
        # Check if figure and axes exist
        if self.fig is None or self.axes is None:
            raise ValueError("Figure and axes not created. Call create_figure() or plot_joint_angles() first.")
        
        # Add normal range to each joint angle plot
        for i, (joint, col) in enumerate(joint_cols.items()):
            if col in mean_data.columns and col in std_data.columns:
                color = colors.get(joint, "k")
                upper = mean_data[col] + std_data[col]
                lower = mean_data[col] - std_data[col]
                self.axes[i].fill_between(mean_data.index, lower, upper, color=color, alpha=alpha, label=f"{joint.capitalize()} Normal Range")
        
        # Update legend
        for ax in self.axes:
            if ax.get_legend() is not None:
                ax.legend()
    
    def save_figure(
        self,
        filename: str,
        dpi: int = None,
        bbox_inches: str = "tight",
        pad_inches: float = 0.1,
        **kwargs
    ) -> None:
        """
        Save the figure to a file.

        Parameters
        ----------
        filename : str
            Filename to save the figure to.
        dpi : int, optional
            Resolution in dots per inch. Default is the dpi used to create the figure.
        bbox_inches : str, optional
            Bounding box in inches. Default is "tight".
        pad_inches : float, optional
            Padding in inches. Default is 0.1.
        **kwargs
            Additional keyword arguments to pass to savefig.
        """
        if self.fig is None:
            raise ValueError("Figure not created. Call create_figure() or plot_joint_angles() first.")
        
        if dpi is None:
            dpi = self.dpi
        
        self.fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs)
    
    def show(self) -> None:
        """
        Show the figure.
        """
        if self.fig is None:
            raise ValueError("Figure not created. Call create_figure() or plot_joint_angles() first.")
        
        plt.show() 