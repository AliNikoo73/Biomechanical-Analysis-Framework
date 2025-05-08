import pytest
import opensim as osim
import numpy as np
from pathlib import Path
import sys
import os

# Add the example directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/basic_moco_example'))

from tracking_simulation import create_tracking_problem, solve_and_visualize

def test_create_tracking_problem():
    """Test the creation of the Moco tracking problem"""
    study, problem = create_tracking_problem()
    
    # Test that study and problem are created correctly
    assert isinstance(study, osim.MocoStudy)
    assert problem is not None
    
    # Test problem parameters
    assert problem.getTimeBounds().getLower() == 0
    assert problem.getTimeBounds().getUpper() == 1.0
    
    # Test model
    model = problem.getModel()
    assert model.getName() == 'pendulum'
    assert model.getNumCoordinates() == 1
    assert model.getNumActuators() == 1

def test_problem_bounds():
    """Test the bounds set in the tracking problem"""
    _, problem = create_tracking_problem()
    
    # Test state bounds
    state_info = problem.getStateInfo('/jointset/ground_pivot/coord0/value')
    assert state_info.getBounds().getLower() == -10
    assert state_info.getBounds().getUpper() == 10
    
    # Test control bounds
    control_info = problem.getControlInfo('/forceset/actuator')
    assert control_info.getBounds().getLower() == -100
    assert control_info.getBounds().getUpper() == 100

@pytest.mark.integration
def test_solution_validity():
    """Test that the solution meets basic validity criteria"""
    study, problem = create_tracking_problem()
    
    # Solve the problem
    solver = study.initCasADiSolver()
    solver.set_num_mesh_intervals(10)  # Reduced for testing
    solution = study.solve()
    
    # Test that solution exists
    assert solution is not None
    
    # Test solution time vector
    time = solution.getTimeMat()
    assert len(time) > 0
    assert time[0] == 0
    assert abs(time[-1] - 1.0) < 1e-6
    
    # Test state trajectory
    states = solution.getStatesTrajectory()
    assert states.shape[1] == 2  # Position and velocity states
    
    # Basic physics check - ensure velocity is derivative of position
    dt = time[1] - time[0]
    velocity_from_position = np.diff(states[:, 0]) / dt
    assert np.allclose(velocity_from_position, states[1:, 1], rtol=0.1)

if __name__ == '__main__':
    pytest.main([__file__]) 