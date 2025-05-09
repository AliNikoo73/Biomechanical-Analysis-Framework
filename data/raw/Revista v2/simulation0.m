%This simulation uses the model without muscles to determine the optimal 
%trajectory between the initial and final points defined by 
%Lee and Umberger (2016).

%To know the work of Lee and Umberger, see:  Lee, L.-F. and Umberger, B. R.
%(2016). Generating optimal control simulations of musculoskeletal 372
%movement using OpenSim and MATLAB. PeerJ, 4:e1638. DOI 10.7717/peerj.1638

%This code was developed by Denis Mosconi. 
%If you have any questions, contact me: denis.mosconi@ifsp.edu.br

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INITIALIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Clearing variables and the screen
clear;
close all;
clc;

%Inclusion of the OpenSim library
import org.opensim.modeling.*

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Opening the LowerLimbA.osim model
OSiminputFilePath = strcat(cd,'\opensimModels\');
OSimModelfileName = 'LowerLimbA.osim';
model = Model([OSiminputFilePath OSimModelfileName]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%OPTIMIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%As the simulation will be predictive, that is, without a reference 
%movement to be tracked, we will use the MocoStudy environment.
study = MocoStudy();

%Optimization environment name
study.setName('sim0');

%Defining the optimal control problem
problem = study.updProblem();

%Adding the model to the problem
problem.setModel(model);

%Setting start time to 0s and end time to 0.75s
iTime = 0;
fTime = 0.75;
problem.setTimeBounds(iTime, fTime);

%Initializing the model
model.initSystem();

%Defining position limits and determining the initial and final positions
%Angles must be indicated in radians.

iHip = deg2rad(-1.414);
fHip = deg2rad(80); 
lBoundH = deg2rad(-5); 
uBoundH = deg2rad(85); 
problem.setStateInfo('/jointset/hip_r/hip_flexion_r/value', ...
    [lBoundH, uBoundH], iHip, fHip);

iKnee = deg2rad(0.927); 
fKnee = deg2rad(-85); 
lBoundK = deg2rad(-90); 
uBoundK = deg2rad(10); 
problem.setStateInfo('/jointset/knee_r/knee_angle_r/value', ...
    [lBoundK, uBoundK], iKnee, fKnee);

iAnkle = deg2rad(-42.773); 
fAnkle = deg2rad(0); 
lBoundA = deg2rad(-45); 
uBoundA = deg2rad(2); 
problem.setStateInfo('/jointset/ankle_r/ankle_angle_r/value', ...
    [lBoundA, uBoundA], iAnkle, fAnkle);

%Setting the start and end speeds to zero
problem.setStateInfoPattern('/jointset/.*/speed', [], 0, 0);

%Adding the objective function to the optimal control problem. 
%In this case, the objective is to minimize the control used to 
%execute the movement.  
effortGoal = MocoControlGoal('controle',100);
problem.addGoal(effortGoal);

%Adding an objective to reduce the sum of squares of states
stateGoal = MocoSumSquaredStateGoal('estados', 0.001);
problem.addGoal(stateGoal);

%Configuring the solver.
solver = study.initCasADiSolver();
solver.set_num_mesh_intervals(100);
solver.set_optim_convergence_tolerance(1e-3);
solver.set_optim_constraint_tolerance(1e-3);

%Solving the problem
tic;
solution = study.solve();
simTime = toc;

%Simulation completed notice.
disp('Simulation executed successfully!');

%Deleting all .asv files
delete *.asv;
























