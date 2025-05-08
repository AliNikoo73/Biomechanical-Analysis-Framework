import unittest
import numpy as np
import opensim as osim
from src.assistive_devices.exoskeleton.ankle_exo import AnkleExoskeleton

class TestAnkleExoskeleton(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.exo = AnkleExoskeleton(
            name="test_exo",
            mass=1.0,
            max_torque=50.0,
            attachment_points={
                'shank': [0.0, -0.2, 0.0],
                'foot': [0.15, 0.02, 0.0]
            }
        )
        
    def test_initialization(self):
        """Test proper initialization of the exoskeleton."""
        self.assertEqual(self.exo.name, "test_exo")
        self.assertEqual(self.exo.mass, 1.0)
        self.assertEqual(self.exo.max_torque, 50.0)
        self.assertIsNotNone(self.exo.attachment_points)
        
    def test_build_device(self):
        """Test device model building."""
        model = self.exo.build_device()
        
        # Check model components
        self.assertIsInstance(model, osim.Model)
        self.assertEqual(model.getName(), "test_exo_model")
        
        # Check actuator creation
        actuators = [comp for comp in model.getComponentsList() 
                    if isinstance(comp, osim.TorqueActuator)]
        self.assertEqual(len(actuators), 1)
        self.assertEqual(actuators[0].getName(), "ankle_assistance")
        
    def test_compute_metrics(self):
        """Test assistance metrics computation."""
        # Create mock simulation results
        simulation_results = {
            'ankle_angle': np.linspace(-20, 20, 100),  # degrees
            'exo_torque': np.sin(np.linspace(0, 2*np.pi, 100)) * 30  # Nm
        }
        
        metrics = self.exo.compute_assistance_metrics(simulation_results)
        
        # Check metric computation
        self.assertIn('peak_assistance_torque', metrics)
        self.assertIn('average_power', metrics)
        self.assertIn('assistance_timing', metrics)
        
        # Check reasonable values
        self.assertLessEqual(metrics['peak_assistance_torque'], self.exo.max_torque)
        self.assertGreaterEqual(metrics['peak_assistance_torque'], 0)
        self.assertGreaterEqual(metrics['average_power'], 0)
        self.assertBetween(metrics['assistance_timing'], 0, 100)
        
    def assertBetween(self, value, low, high):
        """Assert that a value is between two bounds."""
        self.assertGreaterEqual(value, low)
        self.assertLessEqual(value, high)

if __name__ == '__main__':
    unittest.main() 