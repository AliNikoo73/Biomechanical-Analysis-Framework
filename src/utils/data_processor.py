import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from typing import List, Dict, Union, Tuple
import opensim as osim

class BiomechanicsDataProcessor:
    """Class for processing biomechanics data."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.sampling_rate = None
        self.data = None
    
    def load_motion_data(self, file_path: str) -> pd.DataFrame:
        """
        Load motion data from various file formats.
        
        Args:
            file_path: Path to the motion data file
        
        Returns:
            DataFrame containing the motion data
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in ['.mot', '.sto']:
            table = osim.TimeSeriesTable(file_path)
            time = np.array(table.getIndependentColumn())
            labels = [table.getColumnLabels()[i] for i in range(table.getNumColumns())]
            data = np.zeros((len(time), len(labels)))
            
            for i, label in enumerate(labels):
                data[:, i] = np.array(table.getDependentColumn(label))
            
            self.data = pd.DataFrame(data, columns=labels, index=time)
            
        elif file_ext == '.c3d':
            # Implement C3D file reading using ezc3d or similar
            raise NotImplementedError("C3D file support coming soon")
            
        elif file_ext == '.csv':
            self.data = pd.read_csv(file_path)
            
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        return self.data
    
    def filter_data(self, 
                    data: np.ndarray,
                    cutoff_freq: float,
                    sampling_rate: float,
                    order: int = 4) -> np.ndarray:
        """
        Apply Butterworth filter to data.
        
        Args:
            data: Data to filter
            cutoff_freq: Cutoff frequency in Hz
            sampling_rate: Sampling rate in Hz
            order: Filter order
        
        Returns:
            Filtered data
        """
        nyquist = sampling_rate / 2
        normalized_cutoff_freq = cutoff_freq / nyquist
        b, a = signal.butter(order, normalized_cutoff_freq, btype='low')
        return signal.filtfilt(b, a, data, axis=0)
    
    def compute_joint_angles(self, 
                           markers: Dict[str, np.ndarray],
                           joint_def: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Compute joint angles from marker positions.
        
        Args:
            markers: Dictionary of marker positions
            joint_def: Dictionary defining joints using marker names
        
        Returns:
            Dictionary of computed joint angles
        """
        joint_angles = {}
        
        for joint_name, marker_names in joint_def.items():
            if len(marker_names) != 3:
                raise ValueError(f"Joint {joint_name} requires exactly 3 markers")
                
            # Get marker positions
            p1 = markers[marker_names[0]]
            p2 = markers[marker_names[1]]
            p3 = markers[marker_names[2]]
            
            # Compute vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Compute angle
            angle = np.arccos(np.sum(v1 * v2, axis=1) / 
                            (np.linalg.norm(v1, axis=1) * 
                             np.linalg.norm(v2, axis=1)))
            
            joint_angles[joint_name] = np.degrees(angle)
        
        return joint_angles
    
    def compute_velocities(self, 
                         positions: np.ndarray,
                         time: np.ndarray) -> np.ndarray:
        """
        Compute velocities from position data.
        
        Args:
            positions: Position data
            time: Time vector
        
        Returns:
            Computed velocities
        """
        return np.gradient(positions, time, axis=0)
    
    def compute_accelerations(self,
                            velocities: np.ndarray,
                            time: np.ndarray) -> np.ndarray:
        """
        Compute accelerations from velocity data.
        
        Args:
            velocities: Velocity data
            time: Time vector
        
        Returns:
            Computed accelerations
        """
        return np.gradient(velocities, time, axis=0)
    
    def normalize_gait_cycle(self,
                           data: np.ndarray,
                           gait_events: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize data to gait cycle.
        
        Args:
            data: Data to normalize
            gait_events: List of frame indices for gait events
        
        Returns:
            Tuple of (normalized data, percent gait cycle)
        """
        # Create percent gait cycle vector
        percent_gait = np.linspace(0, 100, num=101)
        
        # Initialize normalized data array
        normalized_data = np.zeros((101, data.shape[1]))
        
        # Normalize each gait cycle
        for i in range(len(gait_events)-1):
            start_idx = gait_events[i]
            end_idx = gait_events[i+1]
            cycle_data = data[start_idx:end_idx]
            
            # Interpolate to 101 points
            for j in range(data.shape[1]):
                normalized_data[:, j] += np.interp(
                    percent_gait,
                    np.linspace(0, 100, num=len(cycle_data)),
                    cycle_data[:, j]
                )
        
        # Average across cycles
        normalized_data /= (len(gait_events)-1)
        
        return normalized_data, percent_gait
    
    def detect_gait_events(self,
                          marker_data: np.ndarray,
                          threshold: float = 0.1) -> List[int]:
        """
        Detect heel strike events in marker data.
        
        Args:
            marker_data: Vertical position of heel marker
            threshold: Threshold for event detection
        
        Returns:
            List of frame indices for heel strike events
        """
        # Find local minima in vertical position
        events = signal.find_peaks(-marker_data, height=-threshold)[0]
        return list(events)
    
    def export_processed_data(self,
                            data: Union[np.ndarray, Dict],
                            file_path: str,
                            headers: List[str] = None):
        """
        Export processed data to file.
        
        Args:
            data: Data to export
            file_path: Path to save the file
            headers: Column headers for the data
        """
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame(data, columns=headers)
        
        df.to_csv(file_path, index=False)
        print(f"Data exported to {file_path}")

def main():
    """Main function to demonstrate data processing capabilities."""
    processor = BiomechanicsDataProcessor()
    
    # Load and process motion data
    data = processor.load_motion_data("data/walking_trial.mot")
    
    # Filter data
    filtered_data = processor.filter_data(
        data.values,
        cutoff_freq=6.0,
        sampling_rate=100.0
    )
    
    # Compute kinematics
    velocities = processor.compute_velocities(filtered_data, data.index.values)
    accelerations = processor.compute_accelerations(velocities, data.index.values)
    
    # Export results
    processor.export_processed_data(
        filtered_data,
        "results/processed_motion.csv",
        headers=data.columns
    )
    
    print("Data processing completed successfully!")

if __name__ == "__main__":
    main() 