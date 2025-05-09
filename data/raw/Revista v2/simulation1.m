%Here we use the results obtained through simulation0 as a guess to perform 
%an approximately tracking type simulation using the model with muscles.

%In order to achieve better results, the model used does not contain 
%coordinate actuators, that is, it only contains muscles. For the complete 
%interaction model (muscles + coordinate actuator) see simulation2.m.

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

%Inclusion of relevant folders
addpath('opensimModels');

%Weights for effort minimization (activation) and status tracking, respectively.
controlEffortWeight = 1;
stateTrackingWeight = 300;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%LOADING THE MODEL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Path to the model
OSiminputFilePath = strcat(cd,'\opensimModels\');
OSimModelfileName = 'LowerLimbM.osim';

%Openint the model
model = Model([OSiminputFilePath OSimModelfileName]);

%Opening the model in ModelProcessor mode, to be able to edit it
modelProcessor = ModelProcessor(model);

%Adjusting the model
modelProcessor.append(ModOpIgnoreTendonCompliance());
% Only valid for DeGrooteFregly2016Muscles.
modelProcessor.append(ModOpIgnorePassiveFiberForcesDGF());
% Only valid for DeGrooteFregly2016Muscles.
modelProcessor.append(ModOpScaleActiveFiberForceCurveWidthDGF(1.0));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%OPENING THE FILE sim0_solution.sto
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tableProcessor = TableProcessor('sim0_solution.sto');
tableProcessor.append(TabOpLowPassFilter(6));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%TRACKING PROBLEM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Criating the tracking environment
track = MocoTrack();

%Naming the environment
track.setName('sim1');

%Setting the model to the tracking environment
track.setModel(modelProcessor);

%Adding input guess to the optimization problem
track.setStatesReference(tableProcessor);

%Adding the weights for the tracking optimization part
track.set_states_global_tracking_weight(stateTrackingWeight);

%This setting allows extra data columns contained in the states
%reference that don't correspond to model coordinates.
track.set_allow_unused_references(true);

% Since there is only coordinate position data in the states references, this
% setting is enabled to fill in the missing coordinate speed data using
% the derivative of splined position data.
track.set_track_reference_position_derivatives(true);

%Using the input states as initial guess
track.set_apply_tracked_states_to_guess(true);

%Start time
track.set_initial_time(0.0);

%Final time
track.set_final_time(0.75);

%Time stamp (mesh interval)
track.set_mesh_interval(0.00375);

%Initializing the tracking environment
study = track.initialize();

%Editable access to the problem
problem = study.updProblem();

%Initially the model was opened through the modelProcessor. Now it will be 
%processed and definitively obtained, to make it usable by the program.
model = modelProcessor.process();
model.initSystem();

% Get a reference to the MocoControlGoal that is added to every MocoTrack
% problem by default and configure its weight.
effort = MocoControlGoal.safeDownCast(problem.updGoal('control_effort'));
effort.setWeight(controlEffortWeight);

%Adding an objective to reduce the sum of squares of states
stateGoal = MocoSumSquaredStateGoal('estados', 2);
problem.addGoal(stateGoal);

%Adding an objective to reduce initial activations
initialActivation = MocoInitialActivationGoal();
problem.addGoal(initialActivation);

%Defining the coordinates limits

lBoundH = deg2rad(-5); 
uBoundH = deg2rad(85); 
problem.setStateInfo('/jointset/hip_r/hip_flexion_r/value', [lBoundH, uBoundH]);

lBoundK = deg2rad(-90);
uBoundK = deg2rad(10); 
problem.setStateInfo('/jointset/knee_r/knee_angle_r/value',[lBoundK, uBoundK]);


lBoundA = deg2rad(-45); 
uBoundA = deg2rad(2); 
problem.setStateInfo('/jointset/ankle_r/ankle_angle_r/value', [lBoundA, uBoundA]);

%Solving the problem
tic;
solution = study.solve();
simTime = toc;

%Simulation completed notice.
disp('Simulation executed successfully!');

%Deleting all .asv files
delete *.asv;




























