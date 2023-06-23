import jax
import functools



class Distance_functions():
    def __init__(self, bound, values_bound, NN):
        self.bound = bound
        self.values_bound = values_bound
        self.NN = NN
        self.stabilize = 1e-5

    @functools.partial(jax.jit, static_argnums=(0,))    
    def phi_seg(self, p, a, b):
        f_seg = lambda p, a, b: (p[0]-a[0])*(b[1]-a[1]) - (p[1]-a[1])*(b[0]-a[0]) + self.stabilize
        t_seg = lambda p, a, b : (1/4 - ((p[0]-((b[0]+a[0]))/2)**2 + (p[1]-((b[1]+a[1]))/2)**2)) + self.stabilize
        phi = jax.numpy.sqrt(f_seg(p, a, b)**2+((jax.numpy.sqrt(t_seg(p,a,b)**2 + f_seg(p,a,b)**4)-t_seg(p,a,b))/2)**2)
        return phi + self.stabilize
    

    @functools.partial(jax.jit, static_argnums=(0,))    
    def w_seg(self, p, a, b):
        w = self.phi_seg(p,a,b)**(-2)/jax.numpy.sum(jax.numpy.array([self.phi_seg(p, self.bound[i,:], self.bound[i+1,:])**(-2) for i in range(self.bound.shape[0]-1)] +
                                                                    [self.phi_seg(p, self.bound[self.bound.shape[0]-1,:], self.bound[0,:])**(-2)]))
        return w


    @functools.partial(jax.jit, static_argnums=(0,))    
    def A_func(self, params, points):

        def A_func_point(p):
            AA_func = jax.numpy.zeros(1).reshape((-1,1))
            for i in range(self.bound.shape[0]-1):
                NN = jax.vmap(functools.partial(jax.jit(self.NN), params))(p.reshape(1,-1)).reshape((-1, 1))
                phi_seg = self.phi_seg(p, self.bound[i,:], self.bound[i+1,:]).reshape((-1, 1))
                w_seg = self.w_seg(p, self.bound[i,:], self.bound[i+1,:]).reshape((-1, 1))

                i_A_seg = w_seg*(self.values_bound[i](p).reshape((-1,1)) + phi_seg*NN)
                AA_func += i_A_seg

            NN = jax.vmap(functools.partial(jax.jit(self.NN), params))(p.reshape(1,-1)).reshape((-1, 1))
            phi_seg = self.phi_seg(p, self.bound[self.bound.shape[0]-1,:], self.bound[0,:]).reshape((-1,1))
            w_seg = self.w_seg(p, self.bound[self.bound.shape[0]-1,:], self.bound[0,:]).reshape((-1,1))
            i_A_seg = w_seg*(self.values_bound[-1](p) + phi_seg*NN)
            AA_func += i_A_seg
            return AA_func
        AA_func = jax.vmap(jax.jit(A_func_point))(points).reshape((-1,1))
        return AA_func
    

    @functools.partial(jax.jit, static_argnums=(0,))    
    def F_func(self, params, points):
        dist_func = lambda p: 1/jax.numpy.sum(jax.numpy.array([1/(self.phi_seg(p, self.bound[i,:], self.bound[i+1,:])) for i in range(self.bound.shape[0]-1)] +
                                                              [1/(self.phi_seg(p, self.bound[self.bound.shape[0]-1,:], self.bound[0,:]))]))  
        def F_func_point(p):
            i_F_seg = 1.0
            for i in range(self.bound.shape[0]-1):
                i_F_seg *= self.phi_seg(p, self.bound[i,:], self.bound[i+1,:])**(2) 
            FF_func = i_F_seg * self.phi_seg(p, self.bound[self.bound.shape[0]-1,:], self.bound[0,:])**(-2)
            return FF_func
        NN = jax.vmap(functools.partial(jax.jit(self.NN), params))(points).reshape((-1, 1))
        FF_func = jax.vmap(jax.jit(dist_func))(points).reshape((-1,1))
        F_output = FF_func*NN
        return F_output    
    

    
class PDE_operators:
    """
        Most common operators used to solve PDEs

    Input:
        A function to compute the respective operator
    """
    def __init__(self, function, gfun = None):
        self.function = function
        if gfun != None:
            self.gfun = gfun


    @functools.partial(jax.jit, static_argnums=(0,))    
    def laplacian_2d(self, params, inputs):
        """
        Compute the laplacian

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- coordinates (x, y)

        Returns
        -------
        laplacian : jax.numpy.ndarray[[batch_size, 1]]
            -- laplacian applied to inputs
        """
        fun = lambda params,x,y: self.function(params, x,y)
        @functools.partial(jax.jit)    
        def action(params,x,y):
            u_xx = jax.jacfwd(jax.jacfwd(fun, 1), 1)(params,x,y)
            u_yy = jax.jacfwd(jax.jacfwd(fun, 2), 2)(params,x,y)
            return u_xx + u_yy
        vec_fun = jax.vmap(action, in_axes = (None, 0, 0))
        laplacian = vec_fun(params, inputs[:,0], inputs[:,1])
        return laplacian.reshape((-1, 1))


    @functools.partial(jax.jit, static_argnums=(0,))    
    def laplacian_3d(self, params, inputs):
        """
        Compute the laplacian

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- coordinates (x, y)

        Returns
        -------
        laplacian : jax.numpy.ndarray[[batch_size, 1]]
            -- laplacian applied to inputs
        """
        fun = lambda params,x,y,z: self.function(params, x,y,z)
        @functools.partial(jax.jit)    
        def action(params,x,y,z):
            u_xx = jax.jacfwd(jax.jacfwd(fun, 1), 1)(params,x,y,z)
            u_yy = jax.jacfwd(jax.jacfwd(fun, 2), 2)(params,x,y,z)
            u_zz = jax.jacfwd(jax.jacfwd(fun, 3), 3)(params,x,y,z)
            return u_xx + u_yy + u_zz
        vec_fun = jax.vmap(action, in_axes = (None, 0, 0, 0))
        laplacian = vec_fun(params, inputs[:,0], inputs[:,1], inputs[:,2])
        return laplacian.reshape((-1, 1))


    @functools.partial(jax.jit, static_argnums=(0,))    
    def grad_f_2d(self, params, inputs):
        """
        Compute the gradient

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- coordinates (x, y)

        Returns
        -------
        gradient : jax.numpy.ndarray[[batch_size, 2]]
            -- gradient applied to inputs
        """

        fun = lambda params, x, y: self.function(params, x, y)

        @functools.partial(jax.jit)    
        def action(params, x, y):               
            u_x = jax.jacfwd(fun, 1)(params, x, y)
            u_y = jax.jacfwd(fun, 2)(params, x, y)
            return jax.numpy.column_stack((u_x, u_y))

        vec_fun = jax.vmap(action, in_axes = (None, 0, 0))
        gradient = vec_fun(params, inputs[:,0], inputs[:,1])
        gradient = gradient.reshape((gradient.shape[0], gradient.shape[2]))

        return gradient


    @functools.partial(jax.jit, static_argnums=(0,))    
    def grad_f_3d(self, params, inputs):
        """
        Compute the gradient

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- coordinates (x, y)

        Returns
        -------
        gradient : jax.numpy.ndarray[[batch_size, 2]]
            -- gradient applied to inputs
        """

        fun = lambda params, x, y, z: self.function(params, x, y, z)

        @functools.partial(jax.jit)    
        def action(params, x, y, z):               
            u_x = jax.jacfwd(fun, 1)(params, x, y, z)
            u_y = jax.jacfwd(fun, 2)(params, x, y, z)
            u_z = jax.jacfwd(fun, 3)(params, x, y, z)
            return jax.numpy.column_stack((u_x, u_y, u_z))

        vec_fun = jax.vmap(action, in_axes = (None, 0, 0, 0))
        gradient = vec_fun(params, inputs[:,0], inputs[:,1], inputs[:,2])
        gradient = gradient.reshape((gradient.shape[0], gradient.shape[2]))

        return gradient


    @functools.partial(jax.jit, static_argnums=(0,))    
    def grad_gfun_grad_f_2d(self, params, inputs):
        """
        Compute the dot product between nabla and (gfun*grad_f_2d)

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, 2]]
            -- coordinates (x, y)

        Returns
        -------
        res : jax.numpy.ndarray[[batch_size, 1]]
            -- numerical values of the dot_product between nabla and (gfun*grad_f_2d) applied to inputs
        """

        funx = lambda params, x, y: self.gfun(x,y)*self.grad_f_2d(params, jax.numpy.column_stack((x, y)))[:,0]
        funy = lambda params, x, y: self.gfun(x,y)*self.grad_f_2d(params, jax.numpy.column_stack((x, y)))[:,1]

        @functools.partial(jax.jit)    
        def action(params, x, y):               
            u_x = jax.jacfwd(funx, 1)(params, x, y)
            u_y = jax.jacfwd(funy, 2)(params, x, y)
            return u_x + u_y

        vec_fun = jax.vmap(action, in_axes = (None, 0, 0))
        res = vec_fun(params, inputs[:,0], inputs[:,1])

        return res.reshape((-1,1))



    @functools.partial(jax.jit, static_argnums=(0,))    
    def df_dt_3d(self, params, inputs):
        """
        Compute the time derivative

        Parameters
        ----------
        params : list of parameters[[w1,b1],...,[wn,bn]]
            -- weights and bias
        inputs : jax.numpy.ndarray[[batch_size, 4]]
            -- coordinates and time  (x, y, z, t)

        Returns
        -------
        res : jax.numpy.ndarray[batch_size, 1]
            -- numerical values of the time derivative applied to inputs
        """

        fun = lambda params, x, y, z, t: self.function(params, x, y, z, t)

        @functools.partial(jax.jit)    
        def action(params, x, y, z, t):               
            u_t = jax.jacfwd(fun, 4)(params, x, y, z, t)
            return u_t

        vec_fun = jax.vmap(action, in_axes = (None, 0, 0, 0, 0))
        res = vec_fun(params, inputs[:,0], inputs[:,1], inputs[:,2], inputs[:,3])

        return res.reshape((-1,1))
