This is the version 2.0 of the simulations with OpenSim Moco and Interaction Model.

The foulder opensimModels contain three lowerlimb models:
LowerLimbA: contains only coordinate actuators
LowerLimbM: contains only muscles
LowerLimbIM: contains muscles and coordinate actuator, this is the interaction model

Following the examples provided along with OpenSim Moco, three simulations were prepared
(simulation0, simulation1, simulation2). 

These simulations were performed in this way, because when using full muscles models 
to perform a direct predictive, good results are not obtained. Thus, according to the 
examples provided with Moco, we first run a predictive one with a model without muscles 
and then use the results as a guess for simulations with full muscles models or 
interaction models (as is the case here).

simulation0: predictive simulation with the model containt only coordinate actuators (LowerLimbA)
The results of this simulation are used as a guess for the simulations with full muscles
models or interaction models.

simulation1: a simulation with a full muscle model (LowerLimbM)

simulation 2: a simulation with an interaction model (LowerLimbIM)

In this 2.0 version we also made improvements, adding terms in the 
cost function for minimizing the squares of states as well as the 
initial activations. However, the cost function can be adapted according
 to your needs, simply by commenting out the lines of code corresponding 
to the new terms entered.

If you have any questions, contact me: denis.mosconi@ifsp.edu.br
