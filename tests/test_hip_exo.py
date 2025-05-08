import unittest
import numpy as np
import opensim as osim
from src.assistive_devices.exoskeleton.hip_exo import HipExoskeleton

class TestHipExoskeleton(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.exo = HipExoskeleton(
            name="test_hip_exo",
            mass=2.0,
            max_torque=70.0,
            bilateral=True,
            attachment_points={
                'pelvis': [0.0, 0.0, 0.0],
                'thigh_r': [0.0, -0.15, 0.0],
                'thigh_l': [0.0, -0.15, 0.0]
            }
        )
        
    def test_initialization(self):
        """Test proper initialization of the hip exoskeleton."""
        self.assertEqual(self.exo.name, "test_hip_exo")
        self.assertEqual(self.exo.mass, 2.0)
        self.assertEqual(self.exo.max_torque, 70.0)
        self.assertTrue(self.exo.bilateral)
        self.assertIsNotNone(self.exo.attachment_points)
        
    def test_build_device(self):
        """Test device model building."""
        model = self.exo.build_device()
        
        # Check model components
        self.assertIsInstance(model, osim.Model)
        self.assertEqual(model.getName(), "test_hip_exo_model")
        
        # Check actuator creation for bilateral device
        actuators = [comp for comp in model.getComponentsList() 
                    if isinstance(comp, osim.TorqueActuator)]
        self.assertEqual(len(actuators), 2)  # Should have right and left actuators
        self.assertEqual(actuators[0].getName(), "hip_assistance_r")
        self.assertEqual(actuators[1].getName(), "hip_assistance_l")
        
    def test_unilateral_configuration(self):
        """Test unilateral device configuration."""
        unilateral_exo = HipExoskeleton(
            name="test_unilateral",
            bilateral=False
        )
        model = unilateral_exo.build_device()
        
        # Check actuator creation for unilateral device
        actuators = [comp for comp in model.getComponentsList() 
                    if isinstance(comp, osim.TorqueActuator)]
        self.assertEqual(len(actuators), 1)  # Should have only right actuator
        self.assertEqual(actuators[0].getName(), "hip_assistance_r")
        
    def test_compute_metrics(self):
        """Test assistance metrics computation."""
        # Create mock simulation results for bilateral device
        simulation_results = {
            'hip_angle_r': np.linspace(-30, 30, 100),  # degrees
            'hip_angle_l': np.linspace(-30, 30, 100),  # degrees
            'exo_torque_r': np.sin(np.linspace(0, 2*np.pi, 100)) * 35,  # Nm
            'exo_torque_l': np.sin(np.linspace(0, 2*np.pi, 100)) * 35   # Nm
        }
        
        metrics = self.exo.compute_assistance_metrics(simulation_results)
        
        # Check metric computation
        self.assertIn('peak_assistance_torque_r', metrics)
        self.assertIn('peak_assistance_torque_l', metrics)
        self.assertIn('average_power_r', metrics)
        self.assertIn('average_power_l', metrics)
        self.assertIn('torque_symmetry', metrics)
        self.assertIn('power_symmetry', metrics)
        
        # Check reasonable values
        self.assertLessEqual(metrics['peak_assistance_torque_r'], self.exo.max_torque)
        self.assertLessEqual(metrics['peak_assistance_torque_l'], self.exo.max_torque)
        self.assertGreaterEqual(metrics['average_power_r'], 0)
        self.assertGreaterEqual(metrics['average_power_l'], 0)
        self.assertBetween(metrics['torque_symmetry'], 0, 1)
        self.assertBetween(metrics['power_symmetry'], 0, 1)
        
    def test_optimization_constraints(self):
        """Test optimization with custom constraints."""
        constraints = {
            'max_torque': 50.0,
            'timing_bounds': [10.0, 90.0],
            'power_limit': 150.0,
            'symmetry_threshold': 0.05
        }
        
        # Mock target motion file
        target_motion = "dummy_motion.mot"
        
        # This should run without errors
        try:
            self.exo.optimize_parameters(
                target_motion=target_motion,
                objective_function="metabolic_cost",
                constraints=constraints
            )
        except Exception as e:
            self.fail(f"Optimization failed with constraints: {str(e)}")
            
    def assertBetween(self, value, low, high):
        """Assert that a value is between two bounds."""
        self.assertGreaterEqual(value, low)
        self.assertLessEqual(value, high)

if __name__ == '__main__':
    unittest.main() 