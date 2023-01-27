# M-norm based loss for training data-driven models
A FEM loss is defined as
$\parallel z_\Theta \parallel_M=\sqrt{z_\Theta^\top Mz_\Theta}$ where $z$ is a vector, $\Theta$ is a set of model parameters, and $M$ is the mass matrix of a discretized partial differential equation.
