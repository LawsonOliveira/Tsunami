# Research project

# Solving the propagation equations of a tsunami using neural networks.

# Context :
CentraleSupélec : Pôle projet 005 formation à la recherche

# Supervisor :
- Frédéric Magoulès

# Contributors :
- Rosenberger Julien
- Oliveira Lima Lawson
- Paun Théodore
- Antier Esteban

# Table of Contents
1. [What is PINN ?](#introduction)
2. [First approach - Lagaris](#first_approach)
3. [Second approach](#second_approach)
4. [Third approach(In progress)](#third_approach)
5. [Repository structure](#repo_structure)
6. [Requirements](#requirements)

# What is PINN ? <a name="introduction"></a>
Physics Informed Neural Network is a method for solving partial differential equations using neural networks and physical contraints. Such contraints are imposed using a loss function, which can be expressed in several ways, depending on the PINN model adopted. In this project we used 3 approachs and we consider the PDE defined as:

$Q(\psi, \Delta \psi, Hess \ \psi, . . . )(x_1,x_2,...,x_n,t) = 0 $, inside $\Omega$

$R(\psi, \Delta \psi, Hess \ \psi, . . . )(x_1,x_2,...,x_n,t) = f(x_1,x_2,...,x_n,t) $, in $\partial \Omega$

# First approach - Time independent <a name="first_approach"></a>
In that case the solution is constructed such that:

$\psi=F \cdot NN+A$

where NN is the neural output, F is any function that is non-zero inside the domain and zero at the boundary and A is a function satisfying the boundary conditions. Thus, the loss function is defined using the L2 norm and the residual of the PDE:

$L = {\lVert Q(\psi, \Delta \psi, Hess \ \psi, . . . ) \rVert}_2$ 

## Problem 6 in the paper Lagaris 1998

PINN solution             |  Absolut error
:-------------------------:|:-------------------------:
![Alt text](./Images_readme/NN_Jax_PDE6_20_0.png?raw=true "Title")  |  ![Alt text](./Images_readme/NN_Jax_PDE6_24_0.png?raw=true "Title")





# Second approach - Time dependent <a name="second_approach"></a>
In that case the solution is the output of the neural network and the loss function is defined as:

$L = L_{in}+L_{bound}+L_{initial}$ 

Where

$L_{in} = \frac{1}{N_{in}}{\lVert Q(\psi, \Delta \psi, Hess \ \psi, . . . ) \rVert}^{2}_2$, inside $\Omega$

$L_{bound} = \frac{1}{N_{bound}}{\lVert Q(\psi, \Delta \psi, Hess \ \psi, . . . ) \rVert}^{2}_2$, in $\partial \Omega$

$L_{initial} = \frac{1}{N_{initial}}{\lVert Q(\psi, \Delta \psi, Hess \ \psi, . . . ) \rVert}^{2}_2$, in $t=0$

## Taylor Green Vortex

PINN solution             |  Squared error
:-------------------------:|:-------------------------:
![Alt text](./Images_readme/taylor_green_vortex_p.gif?raw=true "Title")  |  ![Alt text](./Images_readme/squared_error_p.gif?raw=true "Title")

## Mild slope equation
PINN solution 2d            |  PINN solution 3d
:-------------------------:|:-------------------------:
![Alt text](./Images_readme/mild_slope_animation.gif?raw=true "Title")  |  ![Alt text](./Images_readme/mild_slope_animation3d.gif?raw=true "Title") 

# Third approach (In progress) <a name="third_approach"></a>

# Repository structure  <a name="repo_structure"></a>
```
├───doc  
│   ├───Bibliography              -> Reference articles
│   └───polynomails_doc           -> Research on polynomials for certain AI methods  
├───data
│   ├───arcachon_bathymetry       -> Medium resolution dataset
│   ├───arcachon_plus_bathymetry  -> High resolution dataset
│   └───Atlantic_bathymetry       -> Low resolution dataset  
├───manuscript
├───processed_data
│   └───Mesh                      -> Mesh for Arcachon basin
└───src
    ├───Deprecated                -> Old code using polynomials for boundary conditions
    ├───Neural_Networks           -> First tests using PINN with Tf, Torch and Jax (time independent)
    └───PINN_V2                   -> Solving Time dependent PDE with Jax
```

# Requirements  <a name="requirements"></a>
- Jax, Optax, Flax, TensorFlow, PyTorch
- PyVista, VTK, Pandas, Pickle, Matplotlib
