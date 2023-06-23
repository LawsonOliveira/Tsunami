import jax
import functools
import optax
from Models.operators import PDE_operators
import numpy
from scipy.optimize import root


########################################################################################
#################################### First approach ####################################
########################################################################################

class PINN_Poisson_first_approach:
    """
    PINN for Poisson equation

    Input:
        The evaluation function of the neural network and the optimizer
    """
    def __init__(self, NN_evaluation, optimizer, regularize = False):
        self.optimizer = optimizer
        self.NN_evaluation = NN_evaluation

        self.operators = PDE_operators(self.spatial_solution2d)
        self.laplacian = self.operators.laplacian_2d

        fun = lambda params, x, y: self.pde_function(params, jax.numpy.column_stack((x,y)))
        fun = functools.partial(jax.jit(fun))
        self.operators = PDE_operators(fun)
        self.grad2d = self.operators.grad_f_2d
        self.reg = regularize


    @functools.partial(jax.jax.jit, static_argnums = (0, ))    
    def spatial_solution2d(self, params, inputX, inputY):
        """
        Compute the solution of the PDE on (params, x, y)

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputX : jax.numpy.array[batch_size, ]
            -- points in the axis x
        inputY : jax.numpy.array[batch_size, ]
            -- points in the axis y

        Returns
        -------
        F_NN_plus_A : jax.numpy.array[batch_size, 1]
            -- PINN solution applied to inputs.
        """
        inputs = jax.numpy.column_stack((inputX, inputY))
        NN = jax.vmap(functools.partial(jax.jit(self.NN_evaluation), params))(inputs).reshape((-1,1))
        F = self.F_function(inputX, inputY)
        A = self.A_function(inputX, inputY)
        F_NN_plus_A = (F*NN + A).reshape((-1,1))
        return F_NN_plus_A


    @functools.partial(jax.jit, static_argnums=(0,))    
    def A_function(self, inputX, inputY):
        """
        Compute A(x, y) on the inputs

        Parameters
        ----------
        inputX : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        inputY : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        A_output : jax.numpy.ndarray[batch_size, 1]
            -- A(x, y) applied to inputs
        """
        A_output = jax.numpy.zeros_like(inputX)
        return A_output.reshape((-1,1))


    @functools.partial(jax.jit, static_argnums=(0,))    
    def F_function(self, inputX, inputY):
        """
        Compute F(x, y) on the inputs

        Parameters
        ----------
        inputX : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        inputY : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        F_output : jax.numpy.ndarray[batch_size, 1]
            -- F(x, y) applied to inputs
        """
        F1=jax.numpy.multiply(jax.numpy.sin(inputX),jax.numpy.sin(inputX-jax.numpy.ones_like(inputX)))
        F2=jax.numpy.multiply(jax.numpy.sin(inputY),jax.numpy.sin(inputY-jax.numpy.ones_like(inputY)))
        F_output = F1*F2
        return F_output.reshape((-1,1))


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def regularisator(self, params, inputs):
        #reg = jax.numpy.max(abs(self.grad2d(params, inputs))**2)
        if self.reg:
            reg_value = jax.numpy.max(jax.numpy.linalg.norm(self.grad2d(params, inputs), axis = 1)**2)
            reg_value = inputs.shape[0]**(-1.5)*reg_value
        else:
            reg_value = 0.0
        return reg_value
    

    @functools.partial(jax.jit, static_argnums=(0,))    
    def exact_function(self, inputs):
        """
        Compute the exact function on the inputs

        Parameters
        ----------
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y)
            
        Returns
        -------
        exact_output : jax.numpy.ndarray[batch_size, 1]
            -- exact output applied to inputs
        """
        exact_output = (2*jax.numpy.pi**2*jax.numpy.sin(jax.numpy.pi*inputs[:,0])*jax.numpy.sin(jax.numpy.pi*inputs[:,1]))
        return exact_output.reshape((-1,1))


    @functools.partial(jax.jit, static_argnums=(0,))    
    def pde_function(self, params, inputs):
        """
        Compute the pde on the inputs

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y)

        Returns
        -------
        pde_value : jax.numpy.ndarray[batch_size, 1]
            -- pde applied to inputs
        """
        residual = self.laplacian(params,inputs)
        return residual


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def loss_function(self, params, inputs):
        """
        Compute the loss

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y) 

        Returns
        -------
        loss_value : float64
            -- loss applied to inputs
        """
        residual = self.pde_function(params, inputs) + self.exact_function(inputs)
        loss_value = (jax.numpy.linalg.norm(residual)**2)/inputs.shape[0] + self.regularisator(params, inputs)
        return loss_value


    @functools.partial(jax.jax.jit, static_argnums = (0, ))    
    def train_step(self, params, opt_state, inputs):
        """
        Make just one step of the training
        
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        opt_state : a tuple given by optax library
            -- state(hystorical) of the gradient descent
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y)

        Returns
        -------
        loss : a float64
            -- loss function applied to inputs
        new_params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias updated
        opt_state : a tuple given by optax
            -- update the state(hystorical) of the gradient descent
        """
        loss, gradient = jax.value_and_grad(self.loss_function)(params, inputs)
        updates, new_opt_state = self.optimizer.update(gradient, opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state


    @functools.partial(jax.jax.jit, static_argnums = (0, ))    
    def analytical_solution(self, x, y):
        """
        Compute the analytical solution

        Parameters
        ----------
        x : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        y : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        analytical : jax.numpy.ndarray[batch_size, 1]
            -- analytical solution applied to inputs
        """
        analytical = jax.numpy.sin(jax.numpy.pi*x)*jax.numpy.sin(jax.numpy.pi*y)
        return analytical.reshape((-1, 1))

    
class PINN_Helmholtz_first_approach:
    """
    PINN for Helmholtz equation

    Input:
        The evaluation function of the neural network and the optimizer
    """
    def __init__(self, NN_evaluation, optimizer, k = 0.5, regularize = False):
        self.NN_evaluation = NN_evaluation
        self.optimizer = optimizer

        self.operators = PDE_operators(self.spatial_solution2d)
        self.laplacian2d = self.operators.laplacian_2d

        self.k_coeff = k  # Wavenumber



        fun = lambda params, x, y: self.pde_function(params, jax.numpy.column_stack((x,y)))
        fun = functools.partial(jax.jit(fun))
        self.operators = PDE_operators(fun)
        self.grad2d = self.operators.grad_f_2d
        self.reg = regularize

    @functools.partial(jax.jit, static_argnums = (0, ))    
    def spatial_solution2d(self, params, inputX, inputY):
        """
        Compute the solution of the PDE on (params, x, y)

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputX : jax.numpy.array[batch_size, ]
            -- points in the axis x
        inputY : jax.numpy.array[batch_size, ]
            -- points in the axis y

        Returns
        -------
        F_NN_plus_A : jax.numpy.array[batch_size, 1]
            -- PINN solution applied to inputs (complex)
        """
        inputs = jax.numpy.column_stack((inputX, inputY))
        NN = jax.vmap(functools.partial(jax.jit(self.NN_evaluation), params))(inputs).reshape((-1, 1))  
        F = self.F_function(inputX,inputY)
        A = self.A_function(inputX,inputY)


        F_NN_plus_A = F*NN + A
        return F_NN_plus_A.reshape((-1, 1))


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def A_function(self, inputX, inputY):
        """
        Compute A(x, y) on the inputs

        Parameters
        ----------
        inputX : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        inputY : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        A_output : jax.numpy.ndarray[batch_size, 1]
            -- A(x, y) applied to inputs
        """
        j_number = jax.lax.complex(0.0,1.0)
        A1 = (1-inputX)*jax.numpy.exp(j_number*self.k_coeff*inputY) + inputX*jax.numpy.exp(j_number*self.k_coeff*(1+inputY))
        A2 = (1-inputY)*(jax.numpy.exp(j_number*self.k_coeff*inputX) - ((1-inputX) + inputX*jax.numpy.exp(j_number*self.k_coeff)))
        A3 = inputY*(jax.numpy.exp(j_number*self.k_coeff*(inputX + 1)) - ((1-inputX)*jax.numpy.exp(j_number*self.k_coeff) + inputX*jax.numpy.exp(j_number*2*self.k_coeff)))
        A_output = A1 + A2 + A3
        return A_output.reshape((-1, 1))


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def F_function(self, inputX, inputY):
        """
        Compute F(x, y) on the inputs

        Parameters
        ----------
        inputX : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        inputY : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        F_output : jax.numpy.ndarray[batch_size, 1]
            -- F(x, y) applied to inputs
        """
        F1 = jax.numpy.multiply(jax.numpy.sin(inputX), jax.numpy.sin(inputX - 1))
        F2 = jax.numpy.multiply(jax.numpy.sin(inputY), jax.numpy.sin(inputY - 1))
        F_output = F1*F2
        return F_output.reshape((-1, 1))


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def regularisator(self, params, inputs):
        #reg = jax.numpy.max(abs(self.grad2d(params, inputs))**2)
        if self.reg:
            reg_value = jax.numpy.max(jax.numpy.linalg.norm(self.grad2d(params, inputs), axis = 1)**2)
            reg_value = inputs.shape[0]**(-1.5)*reg_value
        else:
            reg_value = 0.0
        return reg_value
    

    @functools.partial(jax.jit, static_argnums = (0, ))    
    def exact_function(self, inputs):
        """
        Compute the exact function on the inputs

        Parameters
        ----------
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y)
            
        Returns
        -------
        exact_output : jax.numpy.ndarray[batch_size, 1]
            -- exact output applied to inputs (complex)
        """
        j_number = jax.lax.complex(0.0,1.0)
        exact_output = self.k_coeff**2*jax.numpy.exp(j_number*self.k_coeff*(inputs[:,0] + inputs[:,1]))
        return exact_output.reshape((-1, 1))

    def update_wave_length(self, k_new):
        """
        Update the wavenumber k

        Parameters
        ----------
        new_k : float
            -- new value of the wavenumber k

        Returns
        -------
        PINN_Helmholtz_first_approach
            -- a new instance of the PINN_Helmholtz_first_approach class with the updated k value
        """
        self.k_coeff = k_new
        return k_new


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def pde_function(self, params, inputs):
        """
        Compute the pde on the inputs

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y)

        Returns
        -------
        pde_value : jax.numpy.ndarray[batch_size, 1]
            -- pde applied to inputs (complex)
        """
        pde_value = self.laplacian2d(params, inputs) + self.k_coeff**2*self.spatial_solution2d(params, inputs[:,0], inputs[:,1])
        return pde_value


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def compute_residual(self, params, inputs):
        """
        Compute the loss

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y) 

        Returns
        -------
        loss_value : float64
            -- loss applied to inputs
        """
        residual = self.pde_function(params, inputs) + self.exact_function(inputs)
        return residual


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def loss_function(self, params, inputs):
        """
        Compute the loss

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y) 

        Returns
        -------
        loss_value : float64
            -- loss applied to inputs
        """
        residual = self.compute_residual(params, inputs)
        loss_value = (jax.numpy.linalg.norm(residual)**2)/inputs.shape[0] + self.regularisator(params, inputs)
        return loss_value


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def train_step(self, params, opt_state, inputs):
        """
        Make just one step of the training

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        opt_state : a tuple given by optax library
            -- state(hystorical) of the gradient descent
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y) 

        Returns
        -------
        loss : float64
            -- loss function applied to inputs
        new_params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias updated
        opt_state : a tuple given by optax
            -- update the state(hystorical) of the gradient descent
        """
        loss, gradient = jax.value_and_grad(self.loss_function)(params, inputs)
        updates, new_opt_state = self.optimizer.update(gradient, opt_state)
        new_params = optax.apply_updates(params, updates)

        return loss, new_params, new_opt_state


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def analytical_solution(self, x, y):
        """
        Compute the analytical solution

        Parameters
        ----------
        x : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        y : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        analytical : jax.numpy.ndarray[batch_size, 1]
            -- analytical solution (complex) applied to input
        """
        j_number = jax.lax.complex(0.0, 1.0)
        sol = jax.numpy.exp(j_number*self.k_coeff*(x + y))
        return sol.reshape((-1, 1))









########################################################################################
#################################### Second approach ###################################
########################################################################################
class PINN_Poisson_second_approach:
    """
    Solve a PDE using Physics Informed Neural Networks   
    Input:
        The evaluation function of the neural network and the optimizer selected to do gradient descent
    """
    def __init__(self, NN_evaluation, optimizer, eta = 1, regularize = False):
        self.optimizer = optimizer
        self.operators = PDE_operators(self.spatial_solution2d)
        self.laplacian = self.operators.laplacian_2d
        self.NN_evaluation = NN_evaluation
        self.eta = eta

        fun = lambda params, x, y: self.pde_function(params, jax.numpy.column_stack((x,y)))
        fun = functools.partial(jax.jit(fun))
        self.operators = PDE_operators(fun)
        self.grad2d = self.operators.grad_f_2d
        self.reg = regularize


    @functools.partial(jax.jax.jit, static_argnums = (0, ))    
    def spatial_solution2d(self, params, inputX, inputY):
        """
        Compute the solution of the PDE on (params, x, y)

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputX : jax.numpy.array[batch_size, ]
            -- points in the axis x
        inputY : jax.numpy.array[batch_size, ]
            -- points in the axis y

        Returns
        -------
        NN : jax.numpy.array[batch_size, 1]
            -- PINN solution applied to inputs
        """
        inputs = jax.numpy.column_stack((inputX, inputY))
        NN = jax.vmap(functools.partial(jax.jit(self.NN_evaluation), params))(inputs)
        return NN.reshape((-1, 1))
    

    @functools.partial(jax.jit, static_argnums=(0,))    
    def pde_function(self, params, inputs):
        """
        Compute the pde on the inputs

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y)

        Returns
        -------
        pde_value : jax.numpy.ndarray[batch_size, 1]
            -- pde applied to inputs
        """
        residual = self.laplacian(params,inputs)
        return residual
    

    @functools.partial(jax.jit, static_argnums = (0, ))    
    def regularisator(self, params, inputs):
        #reg = jax.numpy.max(abs(self.grad2d(params, inputs))**2)
        if self.reg:
            reg_value = jax.numpy.max(jax.numpy.linalg.norm(self.grad2d(params, inputs), axis = 1)**2)
            reg_value = inputs.shape[0]**(-1.5)*reg_value
        else:
            reg_value = 0.0
        return reg_value
    


    @functools.partial(jax.jit, static_argnums=(0,))    
    def loss_boundary(self, params, inputs, values):      
        """
        Compute the loss function at the boundary

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[boundary_size, 2]]
            -- boundary points (x, y)
        values : jax.numpy.ndarray[[boundary_size, 1]]
            -- values of the couple (x, y) at the boundary

        Returns
        -------
        loss_bound : float64
            -- loss function applied to boundary
        """
        exact_bound = values
        preds_bound = self.spatial_solution2d(params, inputs[:,0], inputs[:,1])
        loss_bound = (jax.numpy.linalg.norm(preds_bound - exact_bound)**2)/inputs.shape[0]

        return loss_bound


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def loss_residual(self, params, inputs):
        """
        Compute the residual of the pde

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[boundary_size, 2]]
            -- inside points (x, y)

        Returns
        -------
        loss_residual : float64
            -- loss function applied to inputs
        """
        exact_output = (2*jax.numpy.pi**2*jax.numpy.sin(jax.numpy.pi*inputs[:,0])*jax.numpy.sin(jax.numpy.pi*inputs[:,1])).reshape((-1,1))
        pde_value = self.pde_function(params, inputs)
        residual = pde_value + exact_output
        loss_res = (jax.numpy.linalg.norm(residual)**2)/inputs.shape[0]

        return loss_res


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def loss_function(self, params, inside_points, boundary_points, boundary_values):
        """
        Compute the weighted sum of each loss function

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inside_points : jax.numpy.ndarray[[inside_size, 2]]
            -- inside points (x,y)
        boundary_points : jax.numpy.ndarray[[boundary_size, 2]]
            -- boundary points (x, y)
        boundary_values : jax.numpy.ndarray[[boundary_size, 1]]
            -- values of the couple (x, y) at the boundary

        Returns
        -------
        loss : float64
            -- weighted loss applied to inputs
        losses : numpy.array(2, )
            -- current values of each loss function  [loss_res, loss_bound]
        """
        loss_res = self.loss_residual(params, inside_points)
        loss_bound = self.loss_boundary(params, boundary_points, boundary_values)
        loss_sum = loss_res + self.eta*loss_bound + self.regularisator(params, inside_points)
        losses = jax.numpy.array([loss_res, loss_bound])

        return loss_sum, losses


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def train_step(self, params, opt_state, inside_points, boundary_points, boundary_values):
        """
        Make just one step of the training

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        opt_state : a tuple given by optax
            -- state(hystorical) of the gradient descent
        inside_points : jax.numpy.ndarray[[inside_size, 2]]
            -- inside points (x, y)
        boundary_points : jax.numpy.ndarray[[boundary_size, 2]]
            -- boundary points (x,y)
        boundary_values : jax.numpy.ndarray[[boundary_size, 1]]
            -- values of the couple (x, y) at the boundary

        Returns
        -------
        loss : a float.64
            -- weighted loss applied to inputs
        new_params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias updated
        opt_state : a tuple given by optax
            -- update the state(hystorical) of the gradient descent
        losses : numpy.array(2, )
            -- current values of each loss function  [loss_res, loss_bound]
        """
        (loss, losses), gradient = jax.value_and_grad(self.loss_function, has_aux=True)(params, inside_points, boundary_points, boundary_values)
        updates, new_opt_state = self.optimizer.update(gradient, opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state, losses


    @functools.partial(jax.jax.jit, static_argnums = (0, ))    
    def analytical_solution(self, x, y):
        """
        Compute the analytical solution

        Parameters
        ----------
        x : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        y : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        analytical : jax.numpy.ndarray[batch_size, 1]
            -- analytical solution applied to inputs
        """
        analytical = jax.numpy.sin(jax.numpy.pi*x)*jax.numpy.sin(jax.numpy.pi*y)
        return analytical.reshape((-1, 1))



class PINN_Helmholtz_second_approach:
    """
    Solve a PDE using Physics Informed Neural Networks   
    Input:
        The evaluation function of the neural network and the optimizer selected to do gradient descent
    """
    def __init__(self, NN_evaluation, optimizer, eta = 1, k = 0.5, regularize = False):
        self.NN_evaluation = NN_evaluation
        self.optimizer = optimizer

        self.operators = PDE_operators(self.spatial_solution2d)
        self.laplacian2d = self.operators.laplacian_2d
        self.eta = eta
        self.k_coeff = k # Wavenumber


        fun = lambda params, x, y: self.pde_function(params, jax.numpy.column_stack((x,y)))
        fun = functools.partial(jax.jit(fun))
        self.operators = PDE_operators(fun)
        self.grad2d = self.operators.grad_f_2d
        self.reg = regularize


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def spatial_solution2d(self, params, inputX, inputY):
        """
        Compute the solution of the PDE on (params, x, y)

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputX : jax.numpy.array[batch_size, ]
            -- points in the axis x
        inputY : jax.numpy.array[batch_size, ]
            -- points in the axis y

        Returns
        -------
        NN : jax.numpy.array[batch_size, 1]
            -- PINN solution applied to inputs (complex)
        """
        inputs = jax.numpy.column_stack((inputX, inputY))
        NN = jax.vmap(functools.partial(jax.jit(self.NN_evaluation), params))(inputs)
        return NN.reshape((-1, 1))
    


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def pde_function(self, params, inputs):
        """
        Compute the pde on the inputs

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y)

        Returns
        -------
        pde_value : jax.numpy.ndarray[batch_size, 1]
            -- pde applied to inputs (complex)
        """
        pde_value = self.laplacian2d(params, inputs) + self.k_coeff**2*self.spatial_solution2d(params, inputs[:,0], inputs[:,1])
        return pde_value
    

    @functools.partial(jax.jit, static_argnums = (0, ))    
    def regularisator(self, params, inputs):
        #reg = jax.numpy.max(abs(self.grad2d(params, inputs))**2)
        if self.reg:
            reg_value = jax.numpy.max(jax.numpy.linalg.norm(self.grad2d(params, inputs), axis = 1)**2)
            reg_value = inputs.shape[0]**(-1.5)*reg_value
        else:
            reg_value = 0.0
        return reg_value
    


    @functools.partial(jax.jit, static_argnums=(0,))    
    def loss_boundary(self, params, inputs, values):      
        """
        Compute the loss function at the boundary

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[boundary_size, 2]]
            -- boundary points (x, y)
        values : jax.numpy.ndarray[[boundary_size, 1]]
            -- values of the couple (x, y) at the boundary (complex)

        Returns
        -------
        loss_bound : float64
            -- loss function applied to boundary
        """
        exact_bound = values
        preds_bound = self.spatial_solution2d(params, inputs[:,0], inputs[:,1])
        loss_bound = (jax.numpy.linalg.norm(preds_bound - exact_bound)**2)/inputs.shape[0]
        return loss_bound


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def loss_residual(self, params, inputs):
        """
        Compute the residual of the pde

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[boundary_size, 2]]
            -- inside points (x, y)

        Returns
        -------
        loss_residual : float64
            -- loss function applied to inputs
        """
        j_number = jax.lax.complex(0.0, 1.0)
        exact_pde_values = self.k_coeff**2*jax.numpy.exp(j_number*self.k_coeff*(inputs[:,0] + inputs[:,1])).reshape((-1, 1))
        pred_pde_values = self.pde_function(params, inputs)
        loss_res = (jax.numpy.linalg.norm(pred_pde_values + exact_pde_values)**2)/inputs.shape[0]
        return loss_res


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def loss_function(self, params, inside_points, boundary_points, boundary_values):
        """
        Compute the weighted sum of each loss function

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inside_points : jax.numpy.ndarray[[inside_size, 2]]
            -- inside points (x,y)
        boundary_points : jax.numpy.ndarray[[boundary_size, 2]]
            -- boundary points (x, y)
        boundary_values : jax.numpy.ndarray[[boundary_size, 1]]
            -- values of the couple (x, y) at the boundary (complex)

        Returns
        -------
        loss : float64
            -- weighted loss applied to inputs
        losses : numpy.array(2, )
            -- current values of each loss function  [loss_res, loss_bound]
        """
        loss_res = self.loss_residual(params, inside_points)
        loss_bound = self.loss_boundary(params, boundary_points, boundary_values)
        loss_sum = loss_res + self.eta*loss_bound + self.regularisator(params, inside_points)
        losses = jax.numpy.array([loss_res, loss_bound])
        return loss_sum, losses


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def train_step(self, params, opt_state, inside_points, boundary_points, boundary_values):
        """
        Make just one step of the training

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        opt_state : a tuple given by optax
            -- state(hystorical) of the gradient descent
        inside_points : jax.numpy.ndarray[[inside_size, 2]]
            -- inside points (x, y)
        boundary_points : jax.numpy.ndarray[[boundary_size, 2]]
            -- boundary points (x,y)
        boundary_values : jax.numpy.ndarray[[boundary_size, 1]]
            -- values of the couple (x, y) at the boundary (complex)

        Returns
        -------
        loss : a float.64
            -- weighted loss applied to inputs
        new_params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias updated
        opt_state : a tuple given by optax
            -- update the state(hystorical) of the gradient descent
        losses : numpy.array(2, )
            -- current values of each loss function  [loss_res, loss_bound]
        """
        (loss, losses), gradient = jax.value_and_grad(self.loss_function, has_aux=True)(params, inside_points, boundary_points, boundary_values)
        updates, new_opt_state = self.optimizer.update(gradient, opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state, losses


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def analytical_solution(self, x, y, k = 0.5):
        """
        Compute the analytical solution

        Parameters
        ----------
        x : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        y : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        analytical : jax.numpy.ndarray[batch_size, 1]
            -- analytical solution (complex) applied to input
        """
        j_number = jax.lax.complex(0.0, 1.0)
        sol = jax.numpy.exp(j_number*k*(x + y))
        return sol.reshape((-1, 1))






class PINN_Mild_slope_second_approach:
    """
    Solve a PDE using Physics Informed Neural Networks   
    Input:
        The evaluation function of the neural network, optimizer and shoal coordinates
    """
    def __init__(self, config, NN_evaluation, XYZ_shoal):
        self.NN_evaluation = NN_evaluation
        self.optimizer = config['optimizer']
        self.eta = config['eta']
        mild_function = Mild_slope_functions(config, spatial_2dsolution = self.spatial_2dsolution, shoal_coordinates = XYZ_shoal)
        self.k_coeff = mild_function.k_coeff
        self.c_coeff = mild_function.c_coeff
        self.cg_coeff = mild_function.cg_coeff
        self.incident_psi = mild_function.incident_psi
        self.elevation = mild_function.elevation
        self.c_times_cg = mild_function.c_times_cg_coeff

        self.operators = PDE_operators(self.spatial_2dsolution, self.c_times_cg)
        self.grad_f_2d = self.operators.grad_f_2d
        self.df_dt_3d = self.operators.df_dt_3d
        self.grad_gfun_grad_f_2d = self.operators.grad_gfun_grad_f_2d


    @functools.partial(jax.jit, static_argnums=(0,))    
    def spatial_2dsolution(self, params, inputX, inputY):
        """
        Compute the solution of the PDE on (params, x, y)

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputX : jax.numpy.array[batch_size, ]
            -- points in the axis x
        inputY : jax.numpy.array[batch_size, ]
            -- points in the axis y

        Returns
        -------
        NN : jax.numpy.array[batch_size, 1]
            -- PINN solution applied to inputs (complex)
        """
        inputs = jax.numpy.column_stack((inputX, inputY))
        NN_output = jax.vmap(functools.partial(jax.jit(self.NN_evaluation), params))(inputs)
        
        return NN_output.reshape((-1,1))


    @functools.partial(jax.jit, static_argnums=(0,))    
    def loss_residual(self, params, inputs):
        """
        Compute the residual of the pde

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[boundary_size, 2]]
            -- inside points (x, y)

        Returns
        -------
        loss_residual : float64
            -- loss function applied to inputs
        """
        pred_loss_res = self.grad_gfun_grad_f_2d(params, inputs) + self.k_coeff(inputs[:,0],inputs[:,1])**2*self.c_coeff(inputs[:,0], inputs[:,1])*self.cg_coeff(inputs[:,0], inputs[:,1])*self.spatial_2dsolution(params, inputs[:,0], inputs[:,1])
        loss_res = jax.numpy.linalg.norm(pred_loss_res)**2/pred_loss_res.shape[0]

        return loss_res


    @functools.partial(jax.jit, static_argnums=(0,))    
    def loss_boundary(self, params, boundary_points_list, boundary_normals_list):
        """
        Compute the loss function at the boundary

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        boundary_points_list : list
            -- [bound_inc, bound_neumann_zero_top, bound_absorb, bound_neumann_zero_down]
        boundary_normals_list : list
            -- [normals_inc, normals_bound_neumann_zero_top, normals_bound_absorb, normals_bound_neumann_zero_down]

        Returns
        -------
        loss_bound : float64
            -- loss function applied to boundary
        """
        j_number = jax.lax.complex(0.0, 1.0)
        boundary_points, unit_normal_vector = boundary_points_list[0], boundary_normals_list[0]
        neumann_incident = jax.numpy.einsum('ij,ij->i', self.grad_f_2d(params, boundary_points), unit_normal_vector).reshape((-1, 1)) - j_number*self.k_coeff(boundary_points[:,0], boundary_points[:,1])*(2*self.incident_psi(boundary_points) - self.spatial_2dsolution(params, boundary_points[:,0], boundary_points[:,1]))
        neumann_incident = jax.numpy.linalg.norm(neumann_incident)**2/neumann_incident.shape[0]

        boundary_points, unit_normal_vector = boundary_points_list[1], boundary_normals_list[1]
        neumann_zero_top = jax.numpy.einsum('ij,ij->i',self.grad_f_2d(params, boundary_points), unit_normal_vector).reshape((-1, 1))
        neumann_zero_top = jax.numpy.linalg.norm(neumann_zero_top)**2/neumann_zero_top.shape[0]

        boundary_points, unit_normal_vector = boundary_points_list[2], boundary_normals_list[2]
        neumann_absorbent = jax.numpy.einsum('ij,ij->i',self.grad_f_2d(params, boundary_points), unit_normal_vector).reshape((-1, 1)) - j_number*self.k_coeff(boundary_points[:,0], boundary_points[:,1])*self.spatial_2dsolution(params, boundary_points[:,0], boundary_points[:,1])
        neumann_absorbent = jax.numpy.linalg.norm(neumann_absorbent)**2/neumann_absorbent.shape[0]

        boundary_points, unit_normal_vector = boundary_points_list[3], boundary_normals_list[3]
        neumann_zero_down = jax.numpy.einsum('ij,ij->i',self.grad_f_2d(params, boundary_points), unit_normal_vector).reshape((-1, 1))
        neumann_zero_down = jax.numpy.linalg.norm(neumann_zero_down)**2/neumann_zero_down.shape[0]


        means = jax.numpy.array([neumann_incident, neumann_zero_top, neumann_absorbent, neumann_zero_down])

        loss_bound = jax.numpy.mean(means)

        return loss_bound


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def loss_function(self, params, inside_points, boundary_points, boundary_normals):
        """
        Compute the weighted sum of each loss function

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inside_points : jax.numpy.ndarray[[inside_size, 2]]
            -- inside points (x,y)
        boundary_points : jax.numpy.ndarray[[boundary_size, 2]]
            -- boundary points (x, y)
        boundary_normals : jax.numpy.ndarray[[boundary_size, 1]]
            -- normals of the couple (x, y) at the boundary

        Returns
        -------
        loss : float64
            -- weighted loss applied to inputs
        losses : numpy.array(2, )
            -- current values of each loss function  [loss_res, loss_bound]
        """
        loss_res = self.loss_residual(params, inside_points)
        loss_bound = self.loss_boundary(params, boundary_points, boundary_normals)
        loss_sum =   loss_res + self.eta*loss_bound 
        losses = jax.numpy.array([loss_res, loss_bound])


        return loss_sum, losses


    @functools.partial(jax.jit, static_argnums=(0,))    
    def train_step(self, params, opt_state, inside_points, boundary_points, boundary_normals):
        """
        Make just one step of the training

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        opt_state : a tuple given by optax
            -- state(hystorical) of the gradient descent
        inside_points : jax.numpy.ndarray[[inside_size, 2]]
            -- inside points (x, y)
        boundary_points : jax.numpy.ndarray[[boundary_size, 2]]
            -- boundary points (x,y)
        boundary_normals : jax.numpy.ndarray[[boundary_size, 1]]
            -- normals of the couple (x, y) at the boundary

        Returns
        -------
        loss : a float.64
            -- weighted loss applied to inputs
        new_params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias updated
        opt_state : a tuple given by optax
            -- update the state(hystorical) of the gradient descent
        losses : numpy.array(2, )
            -- current values of each loss function  [loss_res, loss_bound]
        """
        (loss, losses), gradient = jax.value_and_grad(self.loss_function, has_aux=True)(params, inside_points, boundary_points, boundary_normals)
        updates, new_opt_state = self.optimizer.update(gradient, opt_state)
        new_params = optax.apply_updates(params, updates)

        return loss, new_params, new_opt_state, losses






class Mild_slope_functions():
    """
        Class with the physical approach of mild slope equation.
    Input:
        Wave vector, spatial 2d solution of the mild-slope equation and shoal coordinates
    """
    
    def __init__(self, config, spatial_2dsolution, shoal_coordinates):
        self.spatial_2dsolution = spatial_2dsolution
        self.shoal_coordinates = shoal_coordinates
        self.incident_height = config['incident_height_normalized']
        self.angle_direction = config['angle_direction']
        self.operators = PDE_operators(self.solution, self.c_times_cg_coeff)
        self.df_dt_3d = self.operators.df_dt_3d

        if 'wave_vector' in config['characteristic']:
          self.wave_vector = config['characteristic']['wave_vector']
          self.k_coeff = self.k_coeff_with_fixed_k
          self.omega_coeff = self.omega_coeff_with_fixed_k
          omegas = self.omega_coeff_with_fixed_k(shoal_coordinates[:,0], shoal_coordinates[:,1])
          print(f"Omega values - mean: {jax.numpy.mean(omegas)}, min: {jax.numpy.min(omegas)}, max: {jax.numpy.max(omegas)}")

        elif 'omega' in config['characteristic']:
          self.omega = config['characteristic']['omega']
          self.wave_vectors = self.compute_wave_vectors(shoal_coordinates[:,:2], self.omega)
          self.k_coeff = self.k_coeff_with_fixed_W
          self.omega_coeff = self.omega_coeff_with_fixed_W
          print(f"Wave vector values - mean: {jax.numpy.mean(self.wave_vectors)}, min: {jax.numpy.min(self.wave_vectors)}, max: {jax.numpy.max(self.wave_vectors)}")

        else:
          print("Characteristic defined incorrectly")


    def compute_wave_vectors(self, data_xy, omega):
      def equation(k, h, w):
          return k * numpy.tanh(k*h) - w**2

      initial_guess = 1.0  # Initial guess for k
      def solve_equation(h, w):
          solution = root(equation, initial_guess, args=(h, w))
          if solution.success:
              return solution.x[0]
          else:
              print("Solution not found for c:", w, h)
              return None

      heights = numpy.array(self.height_function(data_xy[:,0], data_xy[:,1])).flatten()
      wave_vectors = jax.numpy.array([solve_equation(heights[i], omega) for i in range(heights.shape[0])]).reshape((-1, 1))
      return wave_vectors



    @functools.partial(jax.jit, static_argnums=(0,))    
    def k_coeff_with_fixed_W(self, x, y):
        """
        Compute the wave vector in the couple (x, y)

        Parameters
        ----------
        x : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        y : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        k : jax.numpy.ndarray[[batch_size, 1]]
            -- wave vector on (x, y)
        """
        fun = lambda tx, ty: self.wave_vectors[jax.numpy.argmin(jax.numpy.sqrt(((tx-self.shoal_coordinates[:,0])**2+(ty-self.shoal_coordinates[:,1])**2)))]
        vec_fun = jax.vmap(jax.jit(fun), in_axes = (0, 0))
        k = -vec_fun(x.reshape(-1), y.reshape(-1))
        return k.reshape((-1, 1))


    @functools.partial(jax.jit, static_argnums=(0,))    
    def omega_coeff_with_fixed_W(self, x, y):
        """
        Compute the oscilation omega on (x, y)

        Parameters
        ----------
        x : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        y : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        omega : jax.numpy.ndarray[[batch_size, 1]]
            -- omega applied on (x,y)
        """
        return self.omega


    @functools.partial(jax.jit, static_argnums=(0,))    
    def k_coeff_with_fixed_k(self, x, y):
        """
        Compute the wave vector in the couple (x, y)

        Parameters
        ----------
        x : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        y : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        k : jax.numpy.ndarray[[batch_size, 1]]
            -- wave vector on (x, y)
        """
        return (jax.numpy.ones_like(x)*self.wave_vector).reshape((-1,1 ))


    @functools.partial(jax.jit, static_argnums=(0,))    
    def omega_coeff_with_fixed_k(self, x, y):
        """
        Compute the oscilation omega on (x, y)

        Parameters
        ----------
        x : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        y : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        omega : jax.numpy.ndarray[[batch_size, 1]]
            -- omega applied on (x,y)
        """
        omega = jax.numpy.sqrt(9.81*self.k_coeff(x,y)*jax.numpy.tanh(self.k_coeff(x,y)*self.height_function(x,y)))
        return omega.reshape((-1, 1))


    @functools.partial(jax.jit, static_argnums=(0,))    
    def c_coeff(self, x, y):
        """
        Compute the coefficient c on (x,y)

        Parameters
        ----------
        x : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        y : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        c : jax.numpy.ndarray[[batch_size, 1]]
            -- Coefficient c on (x,y)
        """
        c = self.omega_coeff(x,y)/self.k_coeff(x,y)
        return c.reshape((-1, 1))


    @functools.partial(jax.jit, static_argnums=(0,))    
    def cg_coeff(self, x, y):
        """
        Compute the coefficient cg on (x,y)

        Parameters
        ----------
        x : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        y : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        cg : jax.numpy.ndarray[[batch_size, 1]]
            -- Coefficient cg on (x,y)
        """
        nprime = 0.5*(1 + 2*self.k_coeff(x,y)*self.height_function(x,y)/jax.numpy.sinh(2*self.k_coeff(x,y)*self.height_function(x,y)))
        cg = nprime*self.omega_coeff(x,y)/self.k_coeff(x,y)
        return cg.reshape((-1, 1))


    @functools.partial(jax.jit, static_argnums=(0,))    
    def c_times_cg_coeff(self, x, y):
        """
        Compute the product between c and cg on (x, y)

        Parameters
        ----------
        x : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        y : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        c_times_cg : jax.numpy.ndarray[[batch_size, 1]]
            -- product between c and cg on (x, y)
        """
        c = self.c_coeff(x, y)
        cg = self.cg_coeff(x, y)
        c_times_cg = c*cg
        return c_times_cg.reshape((-1, 1))


    @functools.partial(jax.jit, static_argnums=(0,))    
    def height_function(self, x, y):
        """
        Compute the height in the couple (x, y)

        Parameters
        ----------
        x : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        y : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        h : jax.numpy.ndarray[[batch_size, 1]]
            -- height in each couple (x, y)
        """
        fun = lambda tx, ty: self.shoal_coordinates[:,2][jax.numpy.argmin(jax.numpy.sqrt(((tx-self.shoal_coordinates[:,0])**2+(ty-self.shoal_coordinates[:,1])**2)))]
        vec_fun = jax.vmap(jax.jit(fun), in_axes = (0, 0))
        h = -vec_fun(x.reshape(-1), y.reshape(-1))
        return h.reshape((-1, 1))


    @functools.partial(jax.jit, static_argnums=(0,))    
    def f_dependence(self, inputX, inputY, inputZ):
        """
        Compute the function fz in the point (x, y, z)

        Parameters
        ----------
        x : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        y : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y
        z : jax.numpy.ndarray[batch_size, ]
            -- points in the axis z

        Returns
        -------
        fz : jax.numpy.ndarray[batch_size, 1]
            -- fz in each point (x, y, z)
        """
        fz = (jax.numpy.cosh(self.k_coeff(inputX, inputY)*(self.height_function(inputX, inputY)+ inputZ))/jax.numpy.cosh(self.k_coeff(inputX, inputY)*self.height_function(inputX, inputY)))
        return fz.reshape((-1, 1))


    @functools.partial(jax.jit, static_argnums=(0,))    
    def incident_psi(self, inputs):
        """
        Compute the incident wave in inputs points

        Parameters
        ----------
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y)

        Returns
        -------
        incident_wave : jax.numpy.ndarray[[batch_size, 1]]
            -- incident_wave applied to inputs
        """
        j_number = jax.lax.complex(0.0, 1.0)
        reflection_coefficient = 0.0
        exp_incident = jax.numpy.exp(j_number*self.k_coeff(inputs[:,0],inputs[:,1])   *   (inputs[:,0]*jax.numpy.cos(self.angle_direction) + inputs[:,1]*jax.numpy.sin(self.angle_direction)).reshape((-1, 1)))
        exp_reflected = reflection_coefficient  *   jax.numpy.exp(j_number*self.k_coeff(inputs[:,0],inputs[:,1])   *   (-inputs[:,0]*jax.numpy.cos(self.angle_direction) + inputs[:,1]*jax.numpy.sin(self.angle_direction)).reshape((-1, 1)))
        incident_wave = self.incident_height*(exp_incident + exp_reflected)
        #incident_wave = (2*self.incident_height*9.81/self.c_coeff(inputs[:,0], inputs[:,1])*jax.numpy.exp(-self.k_coeff(inputs[:,0],inputs[:,1])*(inputs[:,0].reshape((-1,1))-0.5)*2*jax.numpy.pi))

        return incident_wave.reshape((-1, 1))
    

    @functools.partial(jax.jit, static_argnums=(0,))    
    def solution(self, params, x,y,z,t):
        """
        Compute the solution at the point (x, y, z, t)

        Parameters
        ----------
        x : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        y : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y
        z : jax.numpy.ndarray[batch_size, ]
            -- points in the axis z
        t : jax.numpy.ndarray[batch_size, ]
            -- points in the time

        Returns
        -------
        res : jax.numpy.ndarray[batch_size, 1]
            -- result in each point (x, y, z, t) (complex)
        """
        j_number = jax.lax.complex(0.0, 1.0)
        complex_time_exp = jax.numpy.exp(-self.omega_coeff(x, y)*t*j_number).reshape((-1, 1))
        res = jax.numpy.real(    self.spatial_2dsolution(params, x, y)    *    self.f_dependence(x,y,z)    *      complex_time_exp)
        return res.reshape((-1, 1))
    


    @functools.partial(jax.jit, static_argnums=(0,))    
    def elevation(self, params, x, y, t):
        """
        Compute the surface displacement in the point (x, y, t)

        Parameters
        ----------
        x : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        y : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y
        t : jax.numpy.ndarray[batch_size, ]
            -- points in the time

        Returns
        -------
        res : jax.numpy.ndarray[batch_size, 1]
            -- Displacement at each point (x, y, z, t)
        """
        inputs = jax.numpy.column_stack((x,y,jax.numpy.zeros_like(y),t))
        res = -self.df_dt_3d(params, inputs)/9.81
        return res.reshape((-1, 1))





class PINN_Helmholtz_Trimesh:
    """
    Solve a PDE using Physics Informed Neural Networks   
    Input:
        The evaluation function of the neural network, optimizer and shoal coordinates
    """
    def __init__(self, config, NN_evaluation):
        self.NN_evaluation = NN_evaluation
        self.optimizer = config['optimizer']
        self.eta = config['eta']
        self.incident_height = config['incident_height']
        self.k_coeff = config['wave_number']

        self.operators = PDE_operators(self.spatial_3dsolution)
        self.grad_f_3d = self.operators.grad_f_3d
        self.laplacian_3d = self.operators.laplacian_3d

    @functools.partial(jax.jit, static_argnums=(0,))    
    def spatial_3dsolution(self, params, inputX, inputY, inputZ):
        """
        Compute the solution of the PDE on (params, x, y)

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputX : jax.numpy.array[batch_size, ]
            -- points in the axis x
        inputY : jax.numpy.array[batch_size, ]
            -- points in the axis y

        Returns
        -------
        NN : jax.numpy.array[batch_size, 1]
            -- PINN solution applied to inputs (complex)
        """
        inputs = jax.numpy.column_stack((inputX, inputY, inputZ))
        NN_output = jax.vmap(functools.partial(jax.jit(self.NN_evaluation), params))(inputs)
        
        return NN_output.reshape((-1,1))


    @functools.partial(jax.jit, static_argnums=(0,))    
    def loss_residual(self, params, inputs):
        """
        Compute the residual of the pde

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[boundary_size, 2]]
            -- inside points (x, y)

        Returns
        -------
        loss_residual : float64
            -- loss function applied to inputs
        """
        j_number = jax.lax.complex(0.0, 1.0)
        #exact_pde_values = self.k_coeff**2*jax.numpy.exp(j_number*self.k_coeff*(inputs[:,0] + inputs[:,1] + inputs[:,2])).reshape((-1, 1))
        exact_pde_values = 0.0
        pred_pde_values = self.laplacian_3d(params, inputs) + self.k_coeff**2*self.spatial_3dsolution(params, inputs[:,0], inputs[:,1], inputs[:,2])
        loss_res = (jax.numpy.linalg.norm(pred_pde_values + exact_pde_values)**2)/inputs.shape[0]
        return loss_res


    @functools.partial(jax.jit, static_argnums=(0,))    
    def loss_boundary(self, params, boundary_points_list, boundary_normals_list):
        """
        Compute the loss function at the boundary

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        boundary_points_list : list
            -- [bound_inc, bound_neumann_zero_top, bound_absorb, bound_neumann_zero_down]
        boundary_normals_list : list
            -- [normals_inc, normals_bound_neumann_zero_top, normals_bound_absorb, normals_bound_neumann_zero_down]

        Returns
        -------
        loss_bound : float64
            -- loss function applied to boundary
        """
        j_number = jax.lax.complex(0.0, 1.0)
        boundary_points, unit_normal_vector = boundary_points_list[0], boundary_normals_list[0]
        neumann_incident = jax.numpy.einsum('ij,ij->i', self.grad_f_3d(params, boundary_points), unit_normal_vector).reshape((-1, 1)) - j_number*self.k_coeff*(2*self.incident_psi(boundary_points) - self.spatial_3dsolution(params, boundary_points[:,0], boundary_points[:,1], boundary_points[:,2]))
        neumann_incident = jax.numpy.linalg.norm(neumann_incident)**2/neumann_incident.shape[0]


        # surface
        boundary_points, unit_normal_vector = boundary_points_list[0], boundary_normals_list[0]
        neumann_zero_surface = jax.numpy.einsum('ij,ij->i',self.grad_f_3d(params, boundary_points), unit_normal_vector).reshape((-1, 1))
        neumann_zero_surface = jax.numpy.linalg.norm(neumann_zero_surface)**2/neumann_zero_surface.shape[0]


        # front
        boundary_points, unit_normal_vector = boundary_points_list[1], boundary_normals_list[1]
        neumann_incident = jax.numpy.einsum('ij,ij->i', self.grad_f_3d(params, boundary_points), unit_normal_vector).reshape((-1, 1)) - j_number*self.k_coeff*(2*self.incident_psi(boundary_points) - self.spatial_3dsolution(params, boundary_points[:,0], boundary_points[:,1], boundary_points[:,2]))
        neumann_incident = jax.numpy.linalg.norm(neumann_incident)**2/neumann_incident.shape[0]


        # behind
        boundary_points, unit_normal_vector = boundary_points_list[2], boundary_normals_list[2]
        neumann_absorbent = jax.numpy.einsum('ij,ij->i',self.grad_f_3d(params, boundary_points), unit_normal_vector).reshape((-1, 1)) - j_number*self.k_coeff*self.spatial_3dsolution(params, boundary_points[:,0], boundary_points[:,1], boundary_points[:,2])
        neumann_absorbent = jax.numpy.linalg.norm(neumann_absorbent)**2/neumann_absorbent.shape[0]


        # left
        boundary_points, unit_normal_vector = boundary_points_list[3], boundary_normals_list[3]
        neumann_zero_left = jax.numpy.einsum('ij,ij->i',self.grad_f_3d(params, boundary_points), unit_normal_vector).reshape((-1, 1))
        neumann_zero_left = jax.numpy.linalg.norm(neumann_zero_left)**2/neumann_zero_left.shape[0]


        # right
        boundary_points, unit_normal_vector = boundary_points_list[4], boundary_normals_list[4]
        neumann_zero_right = jax.numpy.einsum('ij,ij->i',self.grad_f_3d(params, boundary_points), unit_normal_vector).reshape((-1, 1))
        neumann_zero_right = jax.numpy.linalg.norm(neumann_zero_right)**2/neumann_zero_right.shape[0]


        # top
        boundary_points, unit_normal_vector = boundary_points_list[5], boundary_normals_list[5]
        neumann_zero_top = jax.numpy.einsum('ij,ij->i',self.grad_f_3d(params, boundary_points), unit_normal_vector).reshape((-1, 1))
        neumann_zero_top = jax.numpy.linalg.norm(neumann_zero_top)**2/neumann_zero_top.shape[0]


        # down
        boundary_points, unit_normal_vector = boundary_points_list[6], boundary_normals_list[6]
        neumann_zero_down = jax.numpy.einsum('ij,ij->i',self.grad_f_3d(params, boundary_points), unit_normal_vector).reshape((-1, 1))
        neumann_zero_down = jax.numpy.linalg.norm(neumann_zero_down)**2/neumann_zero_down.shape[0]

        means = jax.numpy.array([neumann_zero_surface, neumann_incident, neumann_absorbent, neumann_zero_left, neumann_zero_right, neumann_zero_top, neumann_zero_down])
        loss_bound = jax.numpy.mean(means)

        return loss_bound



    @functools.partial(jax.jit, static_argnums=(0,))    
    def incident_psi(self, inputs):
        """
        Compute the incident wave in inputs points

        Parameters
        ----------
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y)

        Returns
        -------
        incident_wave : jax.numpy.ndarray[[batch_size, 1]]
            -- incident_wave applied to inputs
        """
        j_number = jax.lax.complex(0.0, 1.0)
        reflection_coefficient = 0.0
        exp_incident = jax.numpy.exp(j_number*self.k_coeff   *   (inputs[:,0]*jax.numpy.cos(self.angle_direction) + inputs[:,1]*jax.numpy.sin(self.angle_direction)).reshape((-1, 1)))
        exp_reflected = reflection_coefficient  *   jax.numpy.exp(j_number*self.k_coeff   *   (-inputs[:,0]*jax.numpy.cos(self.angle_direction) + inputs[:,1]*jax.numpy.sin(self.angle_direction)).reshape((-1, 1)))
        incident_wave = self.incident_height*(exp_incident + exp_reflected)
        #incident_wave = (2*self.incident_height*9.81/self.c_coeff(inputs[:,0], inputs[:,1])*jax.numpy.exp(-self.k_coeff(inputs[:,0],inputs[:,1])*(inputs[:,0].reshape((-1,1))-0.5)*2*jax.numpy.pi))

        return incident_wave.reshape((-1, 1))


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def loss_function(self, params, inside_points, boundary_points, boundary_normals):
        """
        Compute the weighted sum of each loss function

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inside_points : jax.numpy.ndarray[[inside_size, 2]]
            -- inside points (x,y)
        boundary_points : jax.numpy.ndarray[[boundary_size, 2]]
            -- boundary points (x, y)
        boundary_normals : jax.numpy.ndarray[[boundary_size, 1]]
            -- normals of the couple (x, y) at the boundary

        Returns
        -------
        loss : float64
            -- weighted loss applied to inputs
        losses : numpy.array(2, )
            -- current values of each loss function  [loss_res, loss_bound]
        """
        loss_res = self.loss_residual(params, inside_points)
        loss_bound = self.loss_boundary(params, boundary_points, boundary_normals)
        loss_sum = loss_res + self.eta*loss_bound
        losses = jax.numpy.array([loss_res, loss_bound])

        return loss_sum, losses


    @functools.partial(jax.jit, static_argnums=(0,))    
    def train_step(self, params, opt_state, inside_points, boundary_points, boundary_normals):
        """
        Make just one step of the training

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        opt_state : a tuple given by optax
            -- state(hystorical) of the gradient descent
        inside_points : jax.numpy.ndarray[[inside_size, 2]]
            -- inside points (x, y)
        boundary_points : jax.numpy.ndarray[[boundary_size, 2]]
            -- boundary points (x,y)
        boundary_normals : jax.numpy.ndarray[[boundary_size, 1]]
            -- normals of the couple (x, y) at the boundary

        Returns
        -------
        loss : a float.64
            -- weighted loss applied to inputs
        new_params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias updated
        opt_state : a tuple given by optax
            -- update the state(hystorical) of the gradient descent
        losses : numpy.array(2, )
            -- current values of each loss function  [loss_res, loss_bound]
        """
        (loss, losses), gradient = jax.value_and_grad(self.loss_function, has_aux=True)(params, inside_points, boundary_points, boundary_normals)
        updates, new_opt_state = self.optimizer.update(gradient, opt_state)
        new_params = optax.apply_updates(params, updates)

        return loss, new_params, new_opt_state, losses










########################################################################################
#################################### Third approach ###################################
########################################################################################
class PINN_Poisson_third_approach:
    """
    PINN for Poisson equation

    Input:
        The evaluation function of the neural network and the optimizer
    """
    def __init__(self, NN_evaluation, optimizer, distance_class, regularize = False):
        self.optimizer = optimizer
        self.NN_evaluation = NN_evaluation

        self.operators = PDE_operators(self.spatial_solution2d)
        self.laplacian = self.operators.laplacian_2d
        self.distance = distance_class



        fun = lambda params, x, y: self.pde_function(params, jax.numpy.column_stack((x,y)))
        fun = functools.partial(jax.jit(fun))
        self.operators = PDE_operators(fun)
        self.grad2d = self.operators.grad_f_2d
        self.reg = regularize

    @functools.partial(jax.jax.jit, static_argnums = (0, ))    
    def spatial_solution2d(self, params, inputX, inputY):
        """
        Compute the solution of the PDE on (params, x, y)

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputX : jax.numpy.array[batch_size, ]
            -- points in the axis x
        inputY : jax.numpy.array[batch_size, ]
            -- points in the axis y

        Returns
        -------
        F_NN_plus_A : jax.numpy.array[batch_size, 1]
            -- PINN solution applied to inputs.
        """
        inputs = jax.numpy.column_stack((inputX, inputY))
        F = self.F_function(params, inputX, inputY)
        A = self.A_function(params, inputX, inputY)
        F_plus_A = (F + A).reshape((-1,1))
        return F_plus_A


    @functools.partial(jax.jit, static_argnums=(0,))    
    def A_function(self, params, inputX, inputY):
        """
        Compute A(x, y) on the inputs

        Parameters
        ----------
        inputX : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        inputY : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        A_output : jax.numpy.ndarray[batch_size, 1]
            -- A(x, y) applied to inputs
        """
        inputs = jax.numpy.column_stack((inputX, inputY))
        A_output = self.distance.A_func(params, inputs)
        return A_output.reshape((-1,1))


    @functools.partial(jax.jit, static_argnums=(0,))    
    def F_function(self, params, inputX, inputY):
        """
        Compute F(x, y) on the inputs

        Parameters
        ----------
        inputX : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        inputY : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        F_output : jax.numpy.ndarray[batch_size, 1]
            -- F(x, y) applied to inputs
        """
        inputs = jax.numpy.column_stack((inputX, inputY))
        F_output = self.distance.F_func(params, inputs)
        return F_output.reshape((-1,1))




    @functools.partial(jax.jit, static_argnums = (0, ))    
    def regularisator(self, params, inputs):
        #reg = jax.numpy.max(abs(self.grad2d(params, inputs))**2)
        if self.reg:
            reg_value = jax.numpy.max(jax.numpy.linalg.norm(self.grad2d(params, inputs), axis = 1)**2)
            reg_value = inputs.shape[0]**(-1.5)*reg_value
        else:
            reg_value = 0.0
        return reg_value
    

    @functools.partial(jax.jit, static_argnums=(0,))    
    def exact_function(self, inputs):
        """
        Compute the exact function on the inputs

        Parameters
        ----------
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y)
            
        Returns
        -------
        exact_output : jax.numpy.ndarray[batch_size, 1]
            -- exact output applied to inputs
        """
        exact_output = (2*jax.numpy.pi**2*jax.numpy.sin(jax.numpy.pi*inputs[:,0])*jax.numpy.sin(jax.numpy.pi*inputs[:,1]))
        return exact_output.reshape((-1,1))


    @functools.partial(jax.jit, static_argnums=(0,))    
    def pde_function(self, params, inputs):
        """
        Compute the pde on the inputs

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y)

        Returns
        -------
        pde_value : jax.numpy.ndarray[batch_size, 1]
            -- pde applied to inputs
        """
        residual = self.laplacian(params,inputs)
        return residual


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def loss_function(self, params, inputs):
        """
        Compute the loss

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y) 

        Returns
        -------
        loss_value : float64
            -- loss applied to inputs
        """
        residual = self.pde_function(params, inputs) + self.exact_function(inputs)
        loss_value = (jax.numpy.linalg.norm(residual)**2)/inputs.shape[0] + self.regularisator(params, inputs)
        return loss_value


    @functools.partial(jax.jax.jit, static_argnums = (0, ))    
    def train_step(self, params, opt_state, inputs):
        """
        Make just one step of the training
        
        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        opt_state : a tuple given by optax library
            -- state(hystorical) of the gradient descent
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y)

        Returns
        -------
        loss : a float64
            -- loss function applied to inputs
        new_params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias updated
        opt_state : a tuple given by optax
            -- update the state(hystorical) of the gradient descent
        """
        loss, gradient = jax.value_and_grad(self.loss_function)(params, inputs)
        updates, new_opt_state = self.optimizer.update(gradient, opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state


    @functools.partial(jax.jax.jit, static_argnums = (0, ))    
    def analytical_solution(self, x, y):
        """
        Compute the analytical solution

        Parameters
        ----------
        x : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        y : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        analytical : jax.numpy.ndarray[batch_size, 1]
            -- analytical solution applied to inputs
        """
        analytical = jax.numpy.sin(jax.numpy.pi*x)*jax.numpy.sin(jax.numpy.pi*y)
        return analytical.reshape((-1, 1))
    




class PINN_Helmholtz_third_approach:
    """
    PINN for Helmholtz equation

    Input:
        The evaluation function of the neural network and the optimizer
    """
    def __init__(self, NN_evaluation, optimizer, distance_class, k = 0.5, regularize = False):
        self.NN_evaluation = NN_evaluation
        self.optimizer = optimizer

        self.operators = PDE_operators(self.spatial_solution2d)
        self.laplacian2d = self.operators.laplacian_2d

        self.k_coeff = k  # Wavenumber
        self.distance  = distance_class

        

        fun = lambda params, x, y: self.pde_function(params, jax.numpy.column_stack((x,y)))
        fun = functools.partial(jax.jit(fun))
        self.operators = PDE_operators(fun)
        self.grad2d = self.operators.grad_f_2d
        self.reg = regularize

    @functools.partial(jax.jax.jit, static_argnums = (0, ))    
    def spatial_solution2d(self, params, inputX, inputY):
        """
        Compute the solution of the PDE on (params, x, y)

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputX : jax.numpy.array[batch_size, ]
            -- points in the axis x
        inputY : jax.numpy.array[batch_size, ]
            -- points in the axis y

        Returns
        -------
        F_NN_plus_A : jax.numpy.array[batch_size, 1]
            -- PINN solution applied to inputs.
        """
        inputs = jax.numpy.column_stack((inputX, inputY))
        F = self.F_function(params, inputX, inputY)
        A = self.A_function(params, inputX, inputY)
        F_plus_A = (F + A).reshape((-1,1))
        return F_plus_A


    @functools.partial(jax.jit, static_argnums=(0,))    
    def A_function(self, params, inputX, inputY):
        """
        Compute A(x, y) on the inputs

        Parameters
        ----------
        inputX : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        inputY : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        A_output : jax.numpy.ndarray[batch_size, 1]
            -- A(x, y) applied to inputs
        """
        inputs = jax.numpy.column_stack((inputX, inputY))
        A_output = self.distance.A_func(params, inputs)
        return A_output.reshape((-1,1))


    @functools.partial(jax.jit, static_argnums=(0,))    
    def F_function(self, params, inputX, inputY):
        """
        Compute F(x, y) on the inputs

        Parameters
        ----------
        inputX : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        inputY : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        F_output : jax.numpy.ndarray[batch_size, 1]
            -- F(x, y) applied to inputs
        """
        inputs = jax.numpy.column_stack((inputX, inputY))
        F_output = self.distance.F_func(params, inputs)
        return F_output.reshape((-1,1))




    @functools.partial(jax.jit, static_argnums = (0, ))    
    def regularisator(self, params, inputs):
        #reg = jax.numpy.max(abs(self.grad2d(params, inputs))**2)
        if self.reg:
            reg_value = jax.numpy.max(jax.numpy.linalg.norm(self.grad2d(params, inputs), axis = 1)**2)
            reg_value = inputs.shape[0]**(-1.5)*reg_value
        else:
            reg_value = 0.0
        return reg_value
    

    @functools.partial(jax.jit, static_argnums = (0, ))    
    def exact_function(self, inputs):
        """
        Compute the exact function on the inputs

        Parameters
        ----------
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y)
            
        Returns
        -------
        exact_output : jax.numpy.ndarray[batch_size, 1]
            -- exact output applied to inputs (complex)
        """
        j_number = jax.lax.complex(0.0,1.0)
        exact_output = self.k_coeff**2*jax.numpy.exp(j_number*self.k_coeff*(inputs[:,0] + inputs[:,1]))
        return exact_output.reshape((-1, 1))


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def pde_function(self, params, inputs):
        """
        Compute the pde on the inputs

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y)

        Returns
        -------
        pde_value : jax.numpy.ndarray[batch_size, 1]
            -- pde applied to inputs (complex)
        """
        pde_value = self.laplacian2d(params, inputs) + self.k_coeff**2*self.spatial_solution2d(params, inputs[:,0], inputs[:,1])
        return pde_value



    @functools.partial(jax.jit, static_argnums = (0, ))    
    def compute_residual(self, params, inputs):
        """
        Compute the loss

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y) 

        Returns
        -------
        loss_value : float64
            -- loss applied to inputs
        """
        residual = self.pde_function(params, inputs) + self.exact_function(inputs)
        return residual


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def loss_function(self, params, inputs):
        """
        Compute the loss

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y) 

        Returns
        -------
        loss_value : float64
            -- loss applied to inputs
        """
        residual = self.pde_function(params, inputs) + self.exact_function(inputs)
        loss_value = (jax.numpy.linalg.norm(residual)**2)/inputs.shape[0] + self.regularisator(params, inputs)
        return loss_value


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def train_step(self, params, opt_state, inputs):
        """
        Make just one step of the training

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        opt_state : a tuple given by optax library
            -- state(hystorical) of the gradient descent
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- points (x, y) 

        Returns
        -------
        loss : float64
            -- loss function applied to inputs
        new_params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias updated
        opt_state : a tuple given by optax
            -- update the state(hystorical) of the gradient descent
        """
        loss, gradient = jax.value_and_grad(self.loss_function)(params, inputs)
        updates, new_opt_state = self.optimizer.update(gradient, opt_state)
        new_params = optax.apply_updates(params, updates)

        return loss, new_params, new_opt_state


    @functools.partial(jax.jit, static_argnums = (0, ))    
    def analytical_solution(self, x, y):
        """
        Compute the analytical solution

        Parameters
        ----------
        x : jax.numpy.ndarray[batch_size, ]
            -- points in the axis x
        y : jax.numpy.ndarray[batch_size, ]
            -- points in the axis y

        Returns
        -------
        analytical : jax.numpy.ndarray[batch_size, 1]
            -- analytical solution (complex) applied to input
        """
        j_number = jax.lax.complex(0.0, 1.0)
        sol = jax.numpy.exp(j_number*self.k_coeff*(x + y))
        return sol.reshape((-1, 1))
