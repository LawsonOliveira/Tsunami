import sympy as sm
import numpy as np

x, y = sm.symbols("x, y")
variables = [x, y]


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
    of the function along with its derivatives in the considered space

    Remarks:
    All expr could be set with numpy. It would simplify the script of this class.
    """

    def __init__(self, coords, strfn):
        """
        Defines the variables used to set F
        - coords: array of coordinates of boundary points (here coords.shape==(nb_points,dimension2))
        - strfn: string relating to the way we set the polynome F
        (among : 'xpy_real', 'xpy_np_real', 'sinxpy_real', 'sinxpy_np_real', 'expxpy_real', 'expxpy_np_real', 'xpy_cmplx', 'xpy_np_cmplx')
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
        # 2 fois la mémoire ? ###############################################################
        set_expr_fn_no_np = self.dic_strfn_to_fn[strfn_no_np]
        self.expr_no_np = set_expr_fn_no_np(coords)
        set_expr_fn = self.dic_strfn_to_fn[strfn]
        self.expr = set_expr_fn(coords)

        # dynamic programming for differentiation
        self.tab_diff = [[self.expr_no_np]]
        self.reduced_tab_diff = []

        self.max_index = 0
        self.dico_order_to_index = {}

    def dynamic_diff_2D(self, l_orders):
        '''
        differentiate F
        Variables:
        - l_orders : list of tuple of size 2

        Returns:
        - reduced_tab_diff : function or list of the expressions of the differentiates of F depending on the strfn of initialization (np or not)
        '''
        for t_order in l_orders:
            (a, b) = t_order
            for i in range(1, a+1):
                if len(self.tab_diff) <= i:
                    self.tab_diff.append([sm.diff(self.tab_diff[i-1][0], x)])
                    print(f'differentiation of order {(i,0)} done ')
            for j in range(1, b+1):
                if len(self.tab_diff[a]) <= j:
                    self.tab_diff[a].append(sm.diff(self.tab_diff[a][j-1], y))
                    print(f'differentiation of order {(j,a)} done ')

        self.dico_order_to_index.update(
            {(a, b): self.max_index+i for i, (a, b) in enumerate(l_orders)})

        if self.is_np_set_expr_fn:
            self.reduced_tab_diff = sm.lambdify(
                variables, [self.tab_diff[a][b] for (a, b) in l_orders], 'numpy')
            return self.reduced_tab_diff

        self.reduced_tab_diff = [self.tab_diff[a][b] for (a, b) in l_orders]
        return self.reduced_tab_diff

    def evaluate(self, X):
        """
        Returns the value of F at coordinates X
        - X: vector of coordinates of a point in 2D. X.shape == (2,) or len(X) == 2

        remark: to see if np could take array instead
        """
        if self.is_np_set_expr_fn:
            return _eval_polynome_numpy(self.expr, X[0], X[1])
        else:
            return _eval_polynome(self.expr, X[0], X[1])

    def evaluate_one_diff(self, t_order, X):
        """
        Returns the value of A at coordinates X
        - t_order: tuple of length len(self.variables) (==2) selecting the derivative
        - X: vector of coordinates of a point in 2D. X.shape == (2,) or len(X) == 2

        Remark: suboptimal with numpy
        """
        (a, b) = t_order
        if t_order in self.dico_order_to_index:
            if self.is_np_set_expr_fn:
                # suboptimal, can evaluate all of them at once with np => done here... and thrown away
                return _eval_polynome_numpy(self.reduced_tab_diff, X[0], X[1])[self.dico_order_to_index[(a, b)]]
            return _eval_polynome(self.reduced_tab_diff[self.dico_order_to_index[(a, b)]], X[0], X[1])
        else:
            dfn = self.dynamic_diff_2D([t_order])
            if self.is_np_set_expr_fn:
                return _eval_polynome_numpy(dfn, X[0], X[1])[0]
            return _eval_polynome(dfn[0], X[0], X[1])


if __name__ == '__main__':
    # dic_strfn_to_fn = {'xpy_real': _set_polynome_xpy_real, 'xpy_np_real': _set_polynome_xpy_numpy_real, 'sinxpy_np_real': _set_polynome_sinxpy_numpy_real,
    #                             'expxpy_np_rl': _set_polynome_expxpy_numpy_real, 'xpy_cmplx': _set_polynome_xpy_cmplx, 'xpy_np_cmplx': _set_polynome_xpy_numpy_cmplx}
    # print(dic_strfn_to_fn)
    # print(dic_strfn_to_fn.keys(),len(dic_strfn_to_fn.keys()))

    # strfn = 'a_b_c'
    # f = strfn.split('_')
    # print(f)
    coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]])
    F = F2D(coords, 'xpy_np_real')
    # print(F.expr(1/2,1/2))
    for X in coords:
        print(F.evaluate(X))
    F.dynamic_diff_2D([(0, 1)])
    print(F.dico_order_to_index)
    for X in coords:
        print(F.evaluate_one_diff((0, 1), X+np.array([1/8, 1/8])))
