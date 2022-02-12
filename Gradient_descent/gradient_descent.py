import sys  
from pathlib import Path  
file = Path(__file__).resolve()  
package_root_directory = file.parents[1]  
sys.path.append(str(package_root_directory))

import numpy as np
import tensorflow as tf
import time 
from Display.display3D_func import _display_surface
from Polynomials.set_coords import _set_coords_square
from Polynomials.polynome4students_v2 import _set_polynome_expxpy_numpy_real

coordinates=_set_coords_square()
t0=time.time()
def _gradient_descent(coords=coordinates):
    # Polynomial definition
    @tf.function
    def set_polynome_expxpy_numpy_real_minimize():
        expr=1
        for i in range(0, coords.shape[0]):
            expr = tf.multiply(1.0 - tf.exp(- 10.0*(x1 - coords[i, 0])**2 - 10.0*(x2 - coords[i, 1])**2),expr)
        return expr

    # Gradient descent
    x1, x2 = tf.Variable(0.0), tf.Variable(0.0)
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    for i in range(250):
        opt.minimize(set_polynome_expxpy_numpy_real_minimize, var_list=[x1, x2])
    t1=time.time()
    return x1,x2,set_polynome_expxpy_numpy_real_minimize(),t1

minx, miny,minz,t1 = _gradient_descent()
print("Xmin :",minx.numpy(),"\nYmin :",miny.numpy(),"\nZmin :",minz.numpy())
print("Gradient descent time(s) : ",t1-t0)

poly = _set_polynome_expxpy_numpy_real(coordinates)
points_min=[minx.numpy(), miny.numpy(),minz.numpy()]
_display_surface(poly,100,1,points_min)
