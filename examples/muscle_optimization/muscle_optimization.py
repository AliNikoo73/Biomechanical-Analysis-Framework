import opensim as osim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

class MuscleOptimizer:
    """Class for muscle force optimization using OpenSim Moco."""
    
    def __init__(self, model_file: str):
        """
        Initialize the muscle optimizer.
        
        Args:
            model_file: Path to the OpenSim model file (.osim)
        """
        self.model = osim.Model(model_file)
        self.model.initSystem()
        
    def setup_moco_study(self, 
                        time_range: tuple,
                        coordinate_bounds: dict,
                        activation_bounds: tuple = (0.01, 1.0)):
        """
        Set up a Moco study for muscle optimization.
        
        Args:
            time_range: Tuple of (start_time, end_time)
            coordinate_bounds: Dictionary of coordinate bounds
            activation_bounds: Tuple of (min_activation, max_activation)
        """
        study = osim.MocoStudy()
        problem = study.initCasADiProblem()
        
        # Set model
        problem.setModel(self.model)
        
        # Set time bounds
        problem.setTimeBounds(time_range[0], time_range[1])
        
        # Set coordinate bounds
        for coord_name, bounds in coordinate_bounds.items():
            problem.setStateInfo(f"/jointset/{coord_name}/value", bounds)
            problem.setStateInfo(f"/jointset/{coord_name}/speed", 
                               [-50, 50])  # Default speed bounds
        
        # Set activation bounds for all muscles
        muscles = self.model.getMuscles()
        for i in range(muscles.getSize()):
            muscle = muscles.get(i)
            problem.setStateInfo(f"/forceset/{muscle.getName()}/activation", 
                               activation_bounds)
        
        return study, problem
    
    def add_effort_goal(self, problem, weight: float = 1.0):
        """
        Add muscle effort minimization goal.
        
        Args:
            problem: MocoProblem instance
            weight: Weight for the effort cost
        """
        effort = osim.MocoControlGoal("effort")
        effort.setWeight(weight)
        problem.addGoal(effort)
    
    def add_tracking_goal(self, 
                         problem, 
                         reference_file: str,
                         weight: float = 1.0):
        """
        Add state tracking goal.
        
        Args:
            problem: MocoProblem instance
            reference_file: Path to reference data file
            weight: Weight for the tracking cost
        """
        tracking = osim.MocoStateTrackingGoal("tracking")
        tracking.setWeight(weight)
        tracking.setReference(reference_file)
        problem.addGoal(tracking)
    
    def solve_optimization(self, study, visualize: bool = True):
        """
        Solve the muscle optimization problem.
        
        Args:
            study: MocoStudy instance
            visualize: Whether to visualize results
        """
        # Configure solver
        solver = study.initCasADiSolver()
        solver.set_num_mesh_intervals(100)
        solver.set_verbosity(2)
        solver.set_optim_solver("ipopt")
        
        # Solve problem
        solution = study.solve()
        
        if visualize:
            self._visualize_results(solution)
        
        return solution
    
    def _visualize_results(self, solution):
        """
        Visualize optimization results.
        
        Args:
            solution: MocoSolution instance
        """
        # Get time and states
        time = solution.getTimeMat()
        states = solution.getStatesTrajectory()
        controls = solution.getControlsTrajectory()
        
        # Plot states
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        for i in range(states.shape[1]):
            plt.plot(time, states[:, i], label=f'State {i+1}')
        plt.xlabel('Time (s)')
        plt.ylabel('State Values')
        plt.title('States Trajectory')
        plt.legend()
        plt.grid(True)
        
        # Plot controls
        plt.subplot(2, 1, 2)
        for i in range(controls.shape[1]):
            plt.plot(time, controls[:, i], label=f'Control {i+1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Control Values')
        plt.title('Controls Trajectory')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('optimization_results.png')
        plt.close()
    
    def export_results(self, solution, output_file: str):
        """
        Export optimization results.
        
        Args:
            solution: MocoSolution instance
            output_file: Path to save results
        """
        # Convert solution to tables
        solution.write(output_file)
        print(f"Results exported to {output_file}")

def main():
    """Main function to demonstrate muscle optimization."""
    # Example usage
    model_file = "models/arm26.osim"
    optimizer = MuscleOptimizer(model_file)
    
    # Setup optimization problem
    coordinate_bounds = {
        "r_shoulder_elev": [-np.pi/2, np.pi/2],
        "r_elbow_flex": [0, np.pi]
    }
    
    study, problem = optimizer.setup_moco_study(
        time_range=(0, 1.0),
        coordinate_bounds=coordinate_bounds
    )
    
    # Add goals
    optimizer.add_effort_goal(problem, weight=1.0)
    optimizer.add_tracking_goal(
        problem,
        reference_file="data/reaching_motion.sto",
        weight=2.0
    )
    
    # Solve and visualize
    solution = optimizer.solve_optimization(study)
    
    # Export results
    optimizer.export_results(solution, "results/optimization_results.sto")
    
    print("Muscle optimization completed successfully!")

if __name__ == "__main__":
    main() 