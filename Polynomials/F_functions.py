import sympy as sm


x, y = sm.symbols("x, y")


def _set_polynome_xpy_real(coords):
    expr = 1
    for i in range(0, coords.shape[0]):
        expr = expr * ((x - coords[i, 0])**2 + (y - coords[i, 1])**2)
    #expr = sm.expand(expr)
    return expr


def _set_polynome_xpy_numpy_real(coords):
    expr = 1
    for i in range(0, coords.shape[0]):
        expr = expr * ((x - coords[i, 0])**2 + (y - coords[i, 1])**2)
    expr = sm.lambdify([x, y], expr, 'numpy')
    return expr


def _set_polynome_sinxpy_real(coords):
    expr = 1
    for i in range(0, coords.shape[0]):
        expr = expr * ((sm.sin(x - coords[i, 0]))
                       ** 2 + (sm.sin(y - coords[i, 1]))**2)
    #expr = sm.expand(expr)
    return expr


def _set_polynome_sinxpy_numpy_real(coords):
    expr = 1
    for i in range(0, coords.shape[0]):
        expr = expr * ((sm.sin(x - coords[i, 0]))
                       ** 2 + (sm.sin(y - coords[i, 1]))**2)
    expr = sm.lambdify([x, y], expr, 'numpy')
    return expr


def _set_polynome_expxpy_real(coords):
    expr = 1
    for i in range(0, coords.shape[0]):
        expr = expr * \
            (1.0 - sm.exp(- 10.0*(x - coords[i, 0])
             ** 2 - 10.0*(y - coords[i, 1])**2))
    #expr = sm.expand(expr)
    return expr


def _set_polynome_expxpy_numpy_real(coords):
    expr = 1
    for i in range(0, coords.shape[0]):
        expr = expr * \
            (1.0 - sm.exp(- 10.0*(x - coords[i, 0])
             ** 2 - 10.0*(y - coords[i, 1])**2))
    expr = sm.lambdify([x, y], expr, 'numpy')
    return expr


def _set_polynome_xpy_cmplx(coords):
    expr = 1
    for i in range(len(coords)):
        expr = expr * (x - coords[i, 0] + 1j*(y - coords[i, 1]))
    #expr = sm.expand(expr)
    return expr


def _set_polynome_xpy_numpy_cmplx(coords):
    expr = 1
    for i in range(len(coords)):
        expr = expr * (x - coords[i, 0] + 1j*(y - coords[i, 1]))
    expr = sm.lambdify([x, y], expr, 'numpy')
    return expr


def _eval_polynome(expr, xi, yi):
    temp = expr.evalf(subs={x: xi, y: yi})
    return temp


def _eval_polynome_numpy(expr, xi, yi):
    temp = expr(xi, yi)
    return temp


"""
Definition of the function F
"""


class F2D():
    """
    Generates the function F on a boundary coords in ###2D### and computes values
    of the function along with its derivatives up to 2nd order in the considered space

    Remarks:
    All expr could be set with numpy. It would simplify the script of this class.
    """

    def __init__(self, coords, strfn):
        """
        Defines the variables used to set F
        - coords: array of coordinates of boundary points (here coords.shape==(nb_points,dimension2))
        - strfn: string relating to the way we set the polynome F
        """
        assert len(coords)
        assert len(coords[0]) == 2
        self.sample = coords

        self.strfn = strfn
        self.dic_strfn_to_fn = {'xpy_real': _set_polynome_xpy_real, 'xpy_np_real': _set_polynome_xpy_numpy_real, 'sinxpy_real': _set_polynome_sinxpy_real, 'sinxpy_np_real': _set_polynome_sinxpy_numpy_real,
                                'expxpy_real': _set_polynome_expxpy_real, 'expxpy_np_real': _set_polynome_expxpy_numpy_real, 'xpy_cmplx': _set_polynome_xpy_cmplx, 'xpy_np_cmplx': _set_polynome_xpy_numpy_cmplx}

        assert strfn in self.dic_strfn_to_fn

        self.variables = [x, y]

        l_str = self.strfn.split('_')
        self.is_np_set_expr_fn = ('np' == l_str[-2])
        if self.is_np_set_expr_fn:
            strfn_no_np = l_str[0]+'_'+l_str[-1]
        else:
            strfn_no_np = strfn
        # 2 fois la mémoire ?
        set_expr_fn_no_np = self.dic_strfn_to_fn[strfn_no_np]
        self.expr_no_np = set_expr_fn_no_np(coords)
        set_expr_fn = self.dic_strfn_to_fn[strfn]
        self.expr = set_expr_fn(coords)

        # dico with keys of length 2 with 1st element the order of derivation wrt x, and 2nd element the order of derivation wrt y
        # A value is the expression of the adequate derivative
        self.dico_derivation = {}

    # length of l_order fixed #####

    def derivate(self, t_order):
        '''
        Return the derivative of F with respect to (variables[i]**l_order[i] for i in range(len(t_order))
        -t_order: tuple of the orders of derivation : we derivate F wrt self.variables[i] t_order[i] times

        Remark : 
        len(l_order)==len(self.variables)        
        use Schwarz theorem
        '''
        assert len(t_order) == len(self.variables)

        if t_order in self.dico_derivation:
            return self.dico_derivation[t_order]
        else:
            # absence d'effet de bord
            dfn = self.expr_no_np
            for i, var in enumerate(self.variables):
                dfn = sm.diff(dfn, var, t_order[i])
            if self.is_np_set_expr_fn:
                dfn = sm.lambdify(self.variables, dfn, 'numpy')
            self.dico_derivation[t_order] = dfn
            return dfn

    def evaluate(self, X):
        """
        Returns the value of F at coordinates X
        - X: vector of coordinates of a point in 2D. X.shape == (2,) or len(X) == 2
        """
        if self.is_np_set_expr_fn:
            return _eval_polynome_numpy(self.expr, X[0], X[1])
        else:
            return _eval_polynome(self.expr, X[0], X[1])

    def evaluate_derivative(self, X, t_order):
        """
        Returns the value of A at coordinates X
        - X: vector of coordinates of a point in 2D. X.shape == (2,) or len(X) == 2
        - t_order: tuple of length len(self.variables) (==2) selecting the derivative
        """
        if t_order in self.dico_derivation:
            if self.is_np_set_expr_fn:
                return _eval_polynome_numpy(self.dico_derivation[t_order], X[0], X[1])
            return _eval_polynome(self.dico_derivation[t_order], X[0], X[1])
        else:
            dfn = self.derivate(t_order)
            if self.is_np_set_expr_fn:
                return _eval_polynome_numpy(dfn, X[0], X[1])
            return _eval_polynome(dfn, X[0], X[1])


if __name__ == '__main__':
    # dic_strfn_to_fn = {'xpy_real': _set_polynome_xpy_real, 'xpy_np_real': _set_polynome_xpy_numpy_real, 'sinxpy_np_real': _set_polynome_sinxpy_numpy_real,
    #                             'expxpy_np_rl': _set_polynome_expxpy_numpy_real, 'xpy_cmplx': _set_polynome_xpy_cmplx, 'xpy_np_cmplx': _set_polynome_xpy_numpy_cmplx}
    # print(dic_strfn_to_fn)
    # print(dic_strfn_to_fn.keys(),len(dic_strfn_to_fn.keys()))

    # strfn = 'a_b_c'
    # f = strfn.split('_')
    # print(f)
    print('ok')
