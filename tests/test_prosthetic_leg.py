import unittest
import numpy as np
import opensim as osim
from src.assistive_devices.prosthetics.prosthetic_leg import ProstheticLeg

class TestProstheticLeg(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.prosthesis = ProstheticLeg(
            name="test_prosthesis",
            mass=3.0,
            amputation_level="transtibial",
            side="right",
            joint_properties={
                'ankle': {
                    'stiffness': 400.0,
                    'damping': 5.0,
                    'range_min': -30.0,
                    'range_max': 30.0
                }
            }
        )
        
    def test_initialization(self):
        """Test proper initialization of the prosthetic leg."""
        self.assertEqual(self.prosthesis.name, "test_prosthesis")
        self.assertEqual(self.prosthesis.mass, 3.0)
        self.assertEqual(self.prosthesis.amputation_level, "transtibial")
        self.assertEqual(self.prosthesis.side, "right")
        self.assertIsNotNone(self.prosthesis.joint_properties)
        
    def test_invalid_configuration(self):
        """Test invalid configuration handling."""
        # Test invalid amputation level
        with self.assertRaises(ValueError):
            ProstheticLeg(name="test", amputation_level="invalid")
            
        # Test invalid side
        with self.assertRaises(ValueError):
            ProstheticLeg(name="test", side="middle")
            
    def test_build_device_transtibial(self):
        """Test building transtibial prosthesis model."""
        model = self.prosthesis.build_device()
        
        # Check model components
        self.assertIsInstance(model, osim.Model)
        self.assertEqual(model.getName(), "test_prosthesis_model")
        
        # Check segment creation
        segments = [comp for comp in model.getComponentsList() 
                   if isinstance(comp, osim.Body)]
        self.assertEqual(len(segments), 2)  # Should have pylon and foot
        
        # Check joint creation
        joints = [comp for comp in model.getComponentsList() 
                 if isinstance(comp, osim.CustomJoint)]
        self.assertEqual(len(joints), 1)  # Should have ankle joint
        
    def test_build_device_transfemoral(self):
        """Test building transfemoral prosthesis model."""
        prosthesis = ProstheticLeg(
            name="test_transfemoral",
            amputation_level="transfemoral",
            joint_properties={
                'knee': {
                    'stiffness': 300.0,
                    'damping': 3.0,
                    'range_min': 0.0,
                    'range_max': 120.0
                },
                'ankle': {
                    'stiffness': 400.0,
                    'damping': 5.0,
                    'range_min': -30.0,
                    'range_max': 30.0
                }
            }
        )
        
        model = prosthesis.build_device()
        
        # Check segment creation
        segments = [comp for comp in model.getComponentsList() 
                   if isinstance(comp, osim.Body)]
        self.assertEqual(len(segments), 3)  # Should have thigh, shank, and foot
        
        # Check joint creation
        joints = [comp for comp in model.getComponentsList() 
                 if isinstance(comp, osim.CustomJoint)]
        self.assertEqual(len(joints), 2)  # Should have knee and ankle joints
        
    def test_compute_metrics(self):
        """Test performance metrics computation."""
        # Create mock simulation results
        simulation_results = {
            'ankle_angle_right': np.linspace(-20, 20, 100),  # degrees
            'ankle_moment_right': np.sin(np.linspace(0, 2*np.pi, 100)) * 30,  # Nm
            'ground_reaction_force': np.column_stack([
                np.zeros(100),  # X force
                np.abs(np.sin(np.linspace(0, np.pi, 100))) * 800,  # Y force
                np.zeros(100)  # Z force
            ])
        }
        
        metrics = self.prosthesis.compute_assistance_metrics(simulation_results)
        
        # Check metric computation
        self.assertIn('ankle_peak_moment', metrics)
        self.assertIn('ankle_range_of_motion', metrics)
        self.assertIn('ankle_average_power', metrics)
        self.assertIn('peak_vertical_force', metrics)
        self.assertIn('loading_rate', metrics)
        self.assertIn('stance_duration', metrics)
        
        # Check reasonable values
        self.assertGreaterEqual(metrics['ankle_range_of_motion'], 0)
        self.assertGreaterEqual(metrics['ankle_average_power'], 0)
        self.assertGreaterEqual(metrics['peak_vertical_force'], 0)
        self.assertGreaterEqual(metrics['loading_rate'], 0)
        self.assertBetween(metrics['stance_duration'], 0, 100)
        
    def test_optimization_constraints(self):
        """Test optimization with custom constraints."""
        constraints = {
            'min_stiffness': 200.0,
            'max_stiffness': 800.0,
            'min_damping': 1.0,
            'max_damping': 10.0
        }
        
        # Mock target motion file
        target_motion = "dummy_motion.mot"
        
        # This should run without errors
        try:
            self.prosthesis.optimize_parameters(
                target_motion=target_motion,
                objective_function="tracking_error",
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