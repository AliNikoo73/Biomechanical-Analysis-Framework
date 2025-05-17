"""
BAF Main Entry Point

This module provides the main entry point for the BAF package when run as a script.
"""

import sys
import argparse
from .gui import launch_app


def main():
    """Main entry point for the BAF package."""
    parser = argparse.ArgumentParser(
        description="Biomechanical Analysis Framework (BAF)"
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # GUI command
    gui_parser = subparsers.add_parser("gui", help="Launch the GUI application")
    
    # Analysis command
    analysis_parser = subparsers.add_parser("analyze", help="Run analysis from command line")
    analysis_parser.add_argument("--input", "-i", required=True, help="Input data file")
    analysis_parser.add_argument("--output", "-o", required=True, help="Output file for results")
    analysis_parser.add_argument("--type", "-t", required=True, 
                              choices=["kinematics", "dynamics", "muscle"],
                              help="Type of analysis to run")
    
    # Visualization command
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    viz_parser.add_argument("--input", "-i", required=True, help="Input data file")
    viz_parser.add_argument("--output", "-o", required=True, help="Output file for visualization")
    viz_parser.add_argument("--type", "-t", required=True, 
                         choices=["joint_angles", "grf", "emg", "combined"],
                         help="Type of visualization to generate")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == "gui" or args.command is None:
        launch_app()
    elif args.command == "analyze":
        print(f"Running {args.type} analysis on {args.input} and saving to {args.output}")
        # TODO: Implement command-line analysis
    elif args.command == "visualize":
        print(f"Generating {args.type} visualization from {args.input} and saving to {args.output}")
        # TODO: Implement command-line visualization
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 