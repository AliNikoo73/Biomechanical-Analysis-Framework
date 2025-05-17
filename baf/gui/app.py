"""
BAF GUI Application

This module provides the main GUI application for the Biomechanical Analysis Framework.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QSplitter,
    QTreeView, QAction, QMenu, QMessageBox, QDockWidget, QComboBox,
    QToolBar, QStatusBar, QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ..visualization.joint_plots import JointPlotter
from ..utils.data_processing import normalize_gait_cycle, detect_gait_events, compute_gait_metrics


class MatplotlibCanvas(FigureCanvas):
    """Canvas for matplotlib figures in the GUI."""
    
    def __init__(self, parent=None, figsize=(5, 4), dpi=100):
        """Initialize the canvas."""
        self.fig = Figure(figsize=figsize, dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        FigureCanvas.setSizePolicy(self, 
                                  QSizePolicy.Expanding, 
                                  QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class MainWindow(QMainWindow):
    """Main window for the BAF GUI application."""
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        
        self.setWindowTitle("Biomechanical Analysis Framework")
        self.setGeometry(100, 100, 1200, 800)
        
        # Data storage
        self.data = {}
        self.current_file = None
        
        # Create the central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create the menu bar
        self.create_menu_bar()
        
        # Create the tool bar
        self.create_tool_bar()
        
        # Create the status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
        # Create the main content area
        self.create_main_content()
        
        # Show the window
        self.show()
    
    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Data", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_data)
        file_menu.addAction(open_action)
        
        save_action = QAction("Save Results", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu("Analysis")
        
        kinematics_action = QAction("Kinematics Analysis", self)
        kinematics_action.triggered.connect(self.run_kinematics_analysis)
        analysis_menu.addAction(kinematics_action)
        
        dynamics_action = QAction("Dynamics Analysis", self)
        dynamics_action.triggered.connect(self.run_dynamics_analysis)
        analysis_menu.addAction(dynamics_action)
        
        muscle_action = QAction("Muscle Analysis", self)
        muscle_action.triggered.connect(self.run_muscle_analysis)
        analysis_menu.addAction(muscle_action)
        
        # Visualization menu
        viz_menu = menubar.addMenu("Visualization")
        
        joint_angles_action = QAction("Joint Angles", self)
        joint_angles_action.triggered.connect(self.visualize_joint_angles)
        viz_menu.addAction(joint_angles_action)
        
        grf_action = QAction("Ground Reaction Forces", self)
        grf_action.triggered.connect(self.visualize_grf)
        viz_menu.addAction(grf_action)
        
        emg_action = QAction("EMG Activity", self)
        emg_action.triggered.connect(self.visualize_emg)
        viz_menu.addAction(emg_action)
        
        # Assistive Devices menu
        devices_menu = menubar.addMenu("Assistive Devices")
        
        exo_action = QAction("Exoskeleton Analysis", self)
        exo_action.triggered.connect(self.analyze_exoskeleton)
        devices_menu.addAction(exo_action)
        
        prosthetic_action = QAction("Prosthetic Analysis", self)
        prosthetic_action.triggered.connect(self.analyze_prosthetic)
        devices_menu.addAction(prosthetic_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        docs_action = QAction("Documentation", self)
        docs_action.triggered.connect(self.open_documentation)
        help_menu.addAction(docs_action)
    
    def create_tool_bar(self):
        """Create the tool bar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)
        
        # Add actions to the toolbar
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_data)
        toolbar.addAction(open_action)
        
        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_results)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        run_action = QAction("Run Analysis", self)
        run_action.triggered.connect(self.run_analysis)
        toolbar.addAction(run_action)
        
        toolbar.addSeparator()
        
        visualize_action = QAction("Visualize", self)
        visualize_action.triggered.connect(self.visualize_results)
        toolbar.addAction(visualize_action)
    
    def create_main_content(self):
        """Create the main content area."""
        # Create a splitter for the main content
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.main_splitter)
        
        # Create the left panel (file browser, etc.)
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        
        # Add a file browser
        self.file_browser_label = QLabel("Data Files:")
        self.left_layout.addWidget(self.file_browser_label)
        
        self.file_tree = QTreeView()
        self.left_layout.addWidget(self.file_tree)
        
        # Add analysis options
        self.analysis_group = QGroupBox("Analysis Options")
        analysis_layout = QFormLayout(self.analysis_group)
        
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems(["Kinematics", "Dynamics", "Muscle Forces", "Optimization"])
        analysis_layout.addRow("Analysis Type:", self.analysis_type_combo)
        
        self.run_analysis_button = QPushButton("Run Analysis")
        self.run_analysis_button.clicked.connect(self.run_analysis)
        analysis_layout.addWidget(self.run_analysis_button)
        
        self.left_layout.addWidget(self.analysis_group)
        
        # Add visualization options
        self.viz_group = QGroupBox("Visualization Options")
        viz_layout = QFormLayout(self.viz_group)
        
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems(["Joint Angles", "Ground Reaction Forces", "Muscle Activity", "Combined"])
        viz_layout.addRow("Plot Type:", self.viz_type_combo)
        
        self.visualize_button = QPushButton("Visualize")
        self.visualize_button.clicked.connect(self.visualize_results)
        viz_layout.addWidget(self.visualize_button)
        
        self.left_layout.addWidget(self.viz_group)
        
        # Add the left panel to the splitter
        self.main_splitter.addWidget(self.left_panel)
        
        # Create the right panel (tabs for different views)
        self.right_panel = QTabWidget()
        
        # Add tabs
        self.data_tab = QWidget()
        self.data_layout = QVBoxLayout(self.data_tab)
        self.data_label = QLabel("No data loaded")
        self.data_layout.addWidget(self.data_label)
        self.right_panel.addTab(self.data_tab, "Data")
        
        self.plot_tab = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_tab)
        self.canvas = MatplotlibCanvas(self.plot_tab, figsize=(5, 4), dpi=100)
        self.plot_layout.addWidget(self.canvas)
        self.plot_toolbar = NavigationToolbar(self.canvas, self.plot_tab)
        self.plot_layout.addWidget(self.plot_toolbar)
        self.right_panel.addTab(self.plot_tab, "Plots")
        
        self.results_tab = QWidget()
        self.results_layout = QVBoxLayout(self.results_tab)
        self.results_label = QLabel("No results available")
        self.results_layout.addWidget(self.results_label)
        self.right_panel.addTab(self.results_tab, "Results")
        
        # Add the right panel to the splitter
        self.main_splitter.addWidget(self.right_panel)
        
        # Set the initial sizes of the splitter
        self.main_splitter.setSizes([300, 900])
    
    def open_data(self):
        """Open data files."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", 
            "All Files (*);;CSV Files (*.csv);;C3D Files (*.c3d)", 
            options=options
        )
        
        if file_name:
            try:
                # Simple CSV loading for now
                if file_name.endswith('.csv'):
                    data = pd.read_csv(file_name)
                    self.data['raw_data'] = data
                    self.current_file = file_name
                    
                    # Update the data view
                    self.data_label.setText(f"Data loaded: {os.path.basename(file_name)}\n"
                                          f"Shape: {data.shape}\n"
                                          f"Columns: {', '.join(data.columns)}")
                    
                    self.statusBar.showMessage(f"Loaded data from {os.path.basename(file_name)}")
                else:
                    QMessageBox.warning(self, "Unsupported File", 
                                      "Currently only CSV files are supported.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
    
    def save_results(self):
        """Save analysis results."""
        if not self.data.get('results'):
            QMessageBox.warning(self, "No Results", "No results available to save.")
            return
        
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", 
            "CSV Files (*.csv);;All Files (*)", 
            options=options
        )
        
        if file_name:
            try:
                # Save results to CSV
                if 'results_df' in self.data:
                    self.data['results_df'].to_csv(file_name, index=False)
                    self.statusBar.showMessage(f"Results saved to {os.path.basename(file_name)}")
                else:
                    # Convert dict to DataFrame if needed
                    results_df = pd.DataFrame.from_dict(self.data['results'], orient='index').T
                    results_df.to_csv(file_name, index=False)
                    self.statusBar.showMessage(f"Results saved to {os.path.basename(file_name)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")
    
    def run_analysis(self):
        """Run the selected analysis."""
        analysis_type = self.analysis_type_combo.currentText()
        
        if not self.data.get('raw_data') is not None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        
        try:
            if analysis_type == "Kinematics":
                self.run_kinematics_analysis()
            elif analysis_type == "Dynamics":
                self.run_dynamics_analysis()
            elif analysis_type == "Muscle Forces":
                self.run_muscle_analysis()
            elif analysis_type == "Optimization":
                QMessageBox.information(self, "Not Implemented", 
                                     "Optimization analysis is not yet implemented.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
    
    def run_kinematics_analysis(self):
        """Run kinematics analysis."""
        if 'raw_data' not in self.data:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        
        # Mock analysis for demonstration
        try:
            # Detect gait events
            # This is a simplified mock implementation
            data = self.data['raw_data']
            
            # Check if the necessary columns exist
            required_cols = ['time', 'hip_angle', 'knee_angle', 'ankle_angle']
            if not all(col in data.columns for col in required_cols):
                QMessageBox.warning(self, "Missing Data", 
                                 f"Data must contain columns: {', '.join(required_cols)}")
                return
            
            # Mock event detection
            events = {
                'foot_strike_1': 0,
                'toe_off': int(len(data) * 0.6),  # Assume toe-off at 60% of data
                'foot_strike_2': len(data) - 1
            }
            
            # Normalize data to gait cycle
            normalized_data = pd.DataFrame(index=np.linspace(0, 100, 101))
            for col in ['hip_angle', 'knee_angle', 'ankle_angle']:
                normalized_data[col] = np.interp(
                    normalized_data.index, 
                    np.linspace(0, 100, len(data)), 
                    data[col].values
                )
            
            # Store results
            self.data['events'] = events
            self.data['normalized_data'] = normalized_data
            
            # Calculate some basic metrics
            metrics = {
                'peak_hip_flexion': normalized_data['hip_angle'].max(),
                'peak_knee_flexion': normalized_data['knee_angle'].max(),
                'peak_ankle_dorsiflexion': normalized_data['ankle_angle'].max(),
                'hip_rom': normalized_data['hip_angle'].max() - normalized_data['hip_angle'].min(),
                'knee_rom': normalized_data['knee_angle'].max() - normalized_data['knee_angle'].min(),
                'ankle_rom': normalized_data['ankle_angle'].max() - normalized_data['ankle_angle'].min(),
            }
            
            self.data['results'] = metrics
            self.data['results_df'] = pd.DataFrame([metrics])
            
            # Update results view
            results_text = "Kinematics Analysis Results:\n\n"
            for key, value in metrics.items():
                results_text += f"{key}: {value:.2f} degrees\n"
            
            self.results_label.setText(results_text)
            
            # Switch to results tab
            self.right_panel.setCurrentIndex(2)
            
            self.statusBar.showMessage("Kinematics analysis completed")
            
            # Automatically visualize the results
            self.visualize_joint_angles()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Kinematics analysis failed: {str(e)}")
    
    def run_dynamics_analysis(self):
        """Run dynamics analysis."""
        QMessageBox.information(self, "Not Implemented", 
                             "Dynamics analysis is not yet fully implemented.")
    
    def run_muscle_analysis(self):
        """Run muscle analysis."""
        QMessageBox.information(self, "Not Implemented", 
                             "Muscle analysis is not yet fully implemented.")
    
    def visualize_results(self):
        """Visualize the selected results."""
        viz_type = self.viz_type_combo.currentText()
        
        try:
            if viz_type == "Joint Angles":
                self.visualize_joint_angles()
            elif viz_type == "Ground Reaction Forces":
                self.visualize_grf()
            elif viz_type == "Muscle Activity":
                self.visualize_emg()
            elif viz_type == "Combined":
                self.visualize_combined()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Visualization failed: {str(e)}")
    
    def visualize_joint_angles(self):
        """Visualize joint angles."""
        if 'normalized_data' not in self.data:
            QMessageBox.warning(self, "No Data", 
                             "Please run kinematics analysis first.")
            return
        
        try:
            # Create a joint plotter
            plotter = JointPlotter(figsize=(8, 4))
            
            # Plot joint angles
            fig, axes = plotter.plot_joint_angles(
                self.data['normalized_data'],
                joint_cols={"hip": "hip_angle", "knee": "knee_angle", "ankle": "ankle_angle"},
                gait_events={"toe_off": 60},  # Assuming toe-off at 60%
                title="Joint Angles During Gait Cycle"
            )
            
            # Update the canvas
            self.canvas.fig.clear()
            for i, ax in enumerate(axes):
                self.canvas.fig.add_subplot(1, 3, i+1)
                self.canvas.fig.axes[i].clear()
                for line in ax.get_lines():
                    self.canvas.fig.axes[i].plot(line.get_xdata(), line.get_ydata(), 
                                              color=line.get_color(), label=line.get_label())
                self.canvas.fig.axes[i].set_title(ax.get_title())
                self.canvas.fig.axes[i].set_xlabel(ax.get_xlabel())
                self.canvas.fig.axes[i].set_ylabel(ax.get_ylabel())
                self.canvas.fig.axes[i].grid(True)
                self.canvas.fig.axes[i].legend()
            
            self.canvas.fig.suptitle("Joint Angles During Gait Cycle", fontsize=14)
            self.canvas.fig.tight_layout()
            self.canvas.draw()
            
            # Switch to plot tab
            self.right_panel.setCurrentIndex(1)
            
            self.statusBar.showMessage("Joint angle visualization completed")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Joint angle visualization failed: {str(e)}")
    
    def visualize_grf(self):
        """Visualize ground reaction forces."""
        QMessageBox.information(self, "Not Implemented", 
                             "GRF visualization is not yet fully implemented.")
    
    def visualize_emg(self):
        """Visualize EMG activity."""
        QMessageBox.information(self, "Not Implemented", 
                             "EMG visualization is not yet fully implemented.")
    
    def visualize_combined(self):
        """Visualize combined results."""
        QMessageBox.information(self, "Not Implemented", 
                             "Combined visualization is not yet fully implemented.")
    
    def analyze_exoskeleton(self):
        """Analyze exoskeleton data."""
        QMessageBox.information(self, "Not Implemented", 
                             "Exoskeleton analysis is not yet fully implemented.")
    
    def analyze_prosthetic(self):
        """Analyze prosthetic data."""
        QMessageBox.information(self, "Not Implemented", 
                             "Prosthetic analysis is not yet fully implemented.")
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, "About BAF",
                        "Biomechanical Analysis Framework (BAF)\n\n"
                        "Version: 0.1.0\n\n"
                        "A comprehensive framework for biomechanical analysis, "
                        "simulation, and assistive device optimization.")
    
    def open_documentation(self):
        """Open documentation."""
        QMessageBox.information(self, "Documentation", 
                             "Documentation will be opened in a web browser.")


def launch_app():
    """Launch the BAF GUI application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())


if __name__ == "__main__":
    launch_app() 