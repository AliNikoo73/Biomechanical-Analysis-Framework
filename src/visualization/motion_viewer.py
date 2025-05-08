import opensim as osim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import vtk
from vtk.util import numpy_support
import pandas as pd

class MotionViewer:
    """Class for visualizing OpenSim motions and results."""
    
    def __init__(self, model_file: str):
        """
        Initialize the motion viewer.
        
        Args:
            model_file: Path to the OpenSim model file (.osim)
        """
        self.model = osim.Model(model_file)
        self.model.initSystem()
        self.state = self.model.initSystem()
        
        # Setup visualization
        self.visualizer = osim.ModelVisualizer(self.model)
        
    def animate_motion(self, motion_file: str, save_video: bool = False):
        """
        Animate a motion file.
        
        Args:
            motion_file: Path to motion file (.mot or .sto)
            save_video: Whether to save animation as video
        """
        # Load motion data
        motion = osim.Storage(motion_file)
        
        # Get time range
        initial_time = motion.getFirstTime()
        final_time = motion.getLastTime()
        
        # Create a states trajectory
        sto = osim.StatesTrajectory.createFromStatesStorage(
            self.model,
            motion,
            [True] * self.model.getNumStateVariables(),
            initial_time,
            final_time
        )
        
        if save_video:
            self._record_animation(sto)
        else:
            self._show_animation(sto)
    
    def plot_muscle_activations(self, states_file: str):
        """
        Plot muscle activation patterns.
        
        Args:
            states_file: Path to states storage file
        """
        # Load states data
        states = osim.Storage(states_file)
        
        # Get muscle names
        muscles = self.model.getMuscles()
        muscle_names = [muscles.get(i).getName() for i in range(muscles.getSize())]
        
        # Extract activation data
        time = np.array(states.getTimeColumn())
        activations = {}
        for muscle in muscle_names:
            activations[muscle] = np.array(states.getDataColumn(
                f"{muscle}_activation"))
        
        # Create plot
        plt.figure(figsize=(12, 8))
        for muscle, activation in activations.items():
            plt.plot(time, activation, label=muscle)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Activation')
        plt.title('Muscle Activation Patterns')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('muscle_activations.png')
        plt.close()
    
    def create_joint_moment_visualization(self, 
                                        forces_file: str,
                                        output_file: str = 'joint_moments.html'):
        """
        Create interactive visualization of joint moments.
        
        Args:
            forces_file: Path to forces storage file
            output_file: Path to save HTML visualization
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Load forces data
        forces = osim.Storage(forces_file)
        time = np.array(forces.getTimeColumn())
        
        # Get coordinate names
        coords = self.model.getCoordinateSet()
        coord_names = [coords.get(i).getName() for i in range(coords.getSize())]
        
        # Extract moment data
        moments = {}
        for coord in coord_names:
            moments[coord] = np.array(forces.getDataColumn(f"{coord}_moment"))
        
        # Create interactive plot
        fig = make_subplots(rows=len(coord_names), cols=1,
                           subplot_titles=coord_names)
        
        for i, (coord, moment) in enumerate(moments.items(), 1):
            fig.add_trace(
                go.Scatter(x=time, y=moment, name=coord),
                row=i, col=1
            )
            
        fig.update_layout(height=300*len(coord_names),
                         title_text="Joint Moments",
                         showlegend=False)
        
        fig.write_html(output_file)
    
    def _show_animation(self, states_trajectory):
        """Show animation in the visualizer window."""
        self.visualizer.showMotion(states_trajectory)
    
    def _record_animation(self, states_trajectory):
        """Record animation to video file."""
        # Setup video writer
        writer = vtk.vtkAVIWriter()
        writer.SetFileName("motion_animation.avi")
        
        # Setup window to image filter
        window = self.visualizer.getCurrentWindow()
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(window)
        
        writer.SetInputConnection(window_to_image.GetOutputPort())
        writer.Start()
        
        # Record frames
        for state in states_trajectory:
            self.visualizer.show(state)
            window_to_image.Modified()
            writer.Write()
        
        writer.End()

def main():
    """Main function to demonstrate visualization capabilities."""
    # Example usage
    model_file = "models/gait2392_simbody.osim"
    viewer = MotionViewer(model_file)
    
    # Animate motion
    viewer.animate_motion("results/walking_motion.mot")
    
    # Plot muscle activations
    viewer.plot_muscle_activations("results/states.sto")
    
    # Create interactive joint moment visualization
    viewer.create_joint_moment_visualization("results/inverse_dynamics.sto")
    
    print("Visualization completed successfully!")

if __name__ == "__main__":
    main() 