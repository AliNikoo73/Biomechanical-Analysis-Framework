import opensim as osim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_tracking_problem():
    """
    Creates a Moco tracking problem for a simple pendulum model.
    """
    # Create a Moco study
    study = osim.MocoStudy()
    
    # Get a model of a simple pendulum
    model = osim.ModelFactory.createPendulum()
    model.setName('pendulum')
    
    # Initialize the problem
    problem = study.initCasADiProblem()
    
    # Define the Moco problem
    problem.setModel(model)
    problem.setTimeBounds(0, 1.0)
    problem.setStateInfo('/jointset/ground_pivot/coord0/value', [-10, 10])
    problem.setStateInfo('/jointset/ground_pivot/coord0/speed', [-50, 50])
    problem.setControlInfo('/forceset/actuator', [-100, 100])

    # Cost: track a reference motion
    tracking = osim.MocoStateTrackingCost('tracking')
    tracking.setWeight(1.0)
    tracking.setReferenceFile('reference_motion.sto')
    problem.addCost(tracking)
    
    return study, problem

def solve_and_visualize(study, problem):
    """
    Solves the Moco problem and visualizes the results.
    """
    # Configure the solver
    solver = study.initCasADiSolver()
    solver.set_num_mesh_intervals(50)
    solver.set_verbosity(2)
    solver.set_optim_solver('ipopt')
    
    # Solve the problem
    solution = study.solve()
    
    # Get the solution states
    time = solution.getTimeMat()
    states = solution.getStatesTrajectory()
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(time, states[:, 0], 'b-', label='Position')
    plt.plot(time, states[:, 1], 'r-', label='Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('State Values')
    plt.legend()
    plt.grid(True)
    plt.title('Pendulum Tracking Results')
    plt.savefig('tracking_results.png')
    plt.close()

def main():
    """
    Main function to run the tracking example.
    """
    print("Creating Moco tracking problem...")
    study, problem = create_tracking_problem()
    
    print("Solving the tracking problem...")
    solve_and_visualize(study, problem)
    
    print("Results have been saved to 'tracking_results.png'")

if __name__ == "__main__":
    main() 