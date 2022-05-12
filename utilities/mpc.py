
# Explanation of MPC

'''
Model predictive control is a high abstraction level method for controlling a plant. It utilizes optimization to predict 
the best input based on a model of the plant in question, defined constraints and knowledge about the current state of the plant.

In practice this is done by defining a cost function that includes the cost of input and the cost of deviations from the desired state, 
and minimizing this for each time step. The model and other factors such as obstructions, physical limitation in 
equipment/plant etc. is added as constraints.

Each solved optimization problem produces the optimal input sequence with the most updated data from the plant. 
Only the first input block is applied to the plant. 
The input for the next timestep will given by the next optimization problem iteration.
'''

'''
Solution:

Model predictive control is a control principle where,
at each timestep, we solve a finite horizon open-loop dynamic optimization
problem using the current state of our system as initial values. The first
input of the resulting input sequence is applied to the plant, and the rest are
discarded. Keywords:
• finite horizon
• open-loop dynamic optimization problem
• apply first input of the sequence

'''