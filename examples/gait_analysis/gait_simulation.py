import opensim as osim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

class GaitAnalyzer:
    """Class for analyzing human gait using OpenSim."""
    
    def __init__(self, model_file: str, motion_file: str):
        """
        Initialize the gait analyzer.
        
        Args:
            model_file: Path to the OpenSim model file (.osim)
            motion_file: Path to the motion file (.mot or .trc)
        """
        self.model = osim.Model(model_file)
        self.model.initSystem()
        self.state = self.model.initSystem()
        self.motion_file = motion_file
        
    def run_inverse_kinematics(self, output_motion_file: str):
        """
        Run inverse kinematics analysis.
        
        Args:
            output_motion_file: Path to save the IK results
        """
        ik_tool = osim.InverseKinematicsTool()
        ik_tool.setModel(self.model)
        ik_tool.setMarkerDataFileName(self.motion_file)
        ik_tool.setOutputMotionFileName(output_motion_file)
        ik_tool.run()
        
        return output_motion_file
    
    def run_inverse_dynamics(self, coordinates_file: str, output_forces_file: str):
        """
        Run inverse dynamics analysis.
        
        Args:
            coordinates_file: Path to coordinates file (from IK)
            output_forces_file: Path to save the ID results
        """
        id_tool = osim.InverseDynamicsTool()
        id_tool.setModel(self.model)
        id_tool.setCoordinatesFileName(coordinates_file)
        id_tool.setOutputGenForceFileName(output_forces_file)
        id_tool.run()
        
        return output_forces_file
    
    def analyze_joint_angles(self, motion_data_file: str):
        """
        Analyze joint angles from motion data.
        
        Args:
            motion_data_file: Path to motion data file
        """
        # Load motion data
        table = osim.TimeSeriesTable(motion_data_file)
        time = np.array(table.getIndependentColumn())
        
        # Get joint angles
        hip_angle = np.array(table.getDependentColumn("hip_flexion_r"))
        knee_angle = np.array(table.getDependentColumn("knee_angle_r"))
        ankle_angle = np.array(table.getDependentColumn("ankle_angle_r"))
        
        # Plot joint angles
        plt.figure(figsize=(12, 6))
        plt.plot(time, hip_angle, 'r-', label='Hip Flexion')
        plt.plot(time, knee_angle, 'b-', label='Knee Angle')
        plt.plot(time, ankle_angle, 'g-', label='Ankle Angle')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.title('Joint Angles During Gait Cycle')
        plt.legend()
        plt.grid(True)
        plt.savefig('joint_angles.png')
        plt.close()
        
        return {
            'time': time,
            'hip_angle': hip_angle,
            'knee_angle': knee_angle,
            'ankle_angle': ankle_angle
        }
    
    def calculate_gait_metrics(self, joint_data: dict):
        """
        Calculate various gait metrics.
        
        Args:
            joint_data: Dictionary containing joint angle data
        """
        metrics = {
            'hip_rom': np.ptp(joint_data['hip_angle']),
            'knee_rom': np.ptp(joint_data['knee_angle']),
            'ankle_rom': np.ptp(joint_data['ankle_angle']),
            'max_hip_flexion': np.max(joint_data['hip_angle']),
            'max_knee_flexion': np.max(joint_data['knee_angle']),
            'max_ankle_dorsiflexion': np.max(joint_data['ankle_angle'])
        }
        
        return pd.DataFrame([metrics])
    
    def export_results(self, metrics: pd.DataFrame, output_file: str):
        """
        Export analysis results to CSV.
        
        Args:
            metrics: DataFrame containing gait metrics
            output_file: Path to save the results
        """
        metrics.to_csv(output_file, index=False)
        print(f"Results exported to {output_file}")

def main():
    """Main function to demonstrate gait analysis."""
    # Example usage
    model_file = "models/gait2392_simbody.osim"
    motion_file = "data/walking_motion.trc"
    
    analyzer = GaitAnalyzer(model_file, motion_file)
    
    # Run analyses
    ik_output = analyzer.run_inverse_kinematics("results/ik_results.mot")
    id_output = analyzer.run_inverse_dynamics(ik_output, "results/id_results.sto")
    
    # Analyze joint angles
    joint_data = analyzer.analyze_joint_angles(ik_output)
    
    # Calculate metrics
    metrics = analyzer.calculate_gait_metrics(joint_data)
    
    # Export results
    analyzer.export_results(metrics, "results/gait_metrics.csv")
    
    print("Gait analysis completed successfully!")

if __name__ == "__main__":
    main() 