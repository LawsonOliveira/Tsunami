import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class SimpleEnvironment():
    def __init__(self, env_bounds, resolution_plot=100):
        self.objects = {}
        self.complete_map = None
        self.env_bounds = env_bounds  # xmin,ymin,xmax,ymax
        self.resolution_plot = resolution_plot

    def _fun_edge(self, vertices_coords, i_node, param_dim, x):
        x0, y0 = vertices_coords[i_node]
        x1, y1 = vertices_coords[(i_node+1) % vertices_coords.shape[0]]
        if param_dim == 0:
            res = (y1-y0)/(x1-x0)*(x-x0)+y0
            return res
        else:
            # param_dim == 1
            res = (x1-x0)/(y1-y0)*(x-y0)+x0
            return res

    def _get_param_dim(self, vertices_coords, i_node, tol=1e-6):
        x0, y0 = vertices_coords[i_node]
        x1, y1 = vertices_coords[(i_node+1) % vertices_coords.shape[0]]
        if np.abs(x1-x0) <= tol:
            assert np.abs(
                y1-y0) > tol, "Two vertices are too close to each other. Consider merging them."
            return 1
        else:
            return 0

    def _half_plane_is_above_edge(self, vertices_coords, i_node, param_dim):
        # if it returns 1, the half plane is above the edge
        i_node_compare = (i_node+2) % vertices_coords.shape[0]
        v_dim_compare = vertices_coords[i_node_compare, param_dim]
        edge_value = self._fun_edge(
            vertices_coords, i_node, param_dim, v_dim_compare)
        if vertices_coords[i_node_compare, (param_dim+1) % 2] >= edge_value:
            return 1
        return 0

    def add_convex_polygon(self, vertices_coords: np.array):
        '''
        Add a convex polygon to the environment
        '''
        assert vertices_coords.shape[0] >= 3, "Not a polygon"

        polygon_id = len(self.objects)
        self.objects[polygon_id] = {
            "type": "polygon",
            "vertices_coords": vertices_coords
        }

        X = np.linspace(
            self.env_bounds[0], self.env_bounds[2], self.resolution_plot)
        Y = np.linspace(
            self.env_bounds[1], self.env_bounds[3], self.resolution_plot)

        xv, yv = np.meshgrid(X, Y)
        map = self._is_in_this_polygon(polygon_id, xv, yv)

        self.objects[polygon_id]["map"] = map

    def add_disc(self, centers_coords: np.array, radius: float):
        '''
        Add a disc to the environment
        '''
        assert centers_coords.shape == (2,)
        disc_id = len(self.objects)
        self.objects[disc_id] = {
            "type": "disc",
            "center_coords": centers_coords,
            "radius": radius
        }

        X = np.linspace(
            self.env_bounds[0], self.env_bounds[2], self.resolution_plot)
        Y = np.linspace(
            self.env_bounds[1], self.env_bounds[3], self.resolution_plot)

        xv, yv = np.meshgrid(X, Y)
        map = self._is_in_this_disc(disc_id, xv, yv)

        self.objects[disc_id]["map"] = map

    def _is_in_this_polygon(self, polygon_id, X, Y):
        vertices_coords = self.objects[polygon_id]["vertices_coords"]

        res = None
        for i_node in range(vertices_coords.shape[0]):
            param_dim = self._get_param_dim(vertices_coords, i_node)
            if self._half_plane_is_above_edge(vertices_coords, i_node, param_dim):
                if param_dim == 0:
                    is_in_half_plane = Y >= self._fun_edge(
                        vertices_coords, i_node, param_dim, X)
                else:
                    is_in_half_plane = X >= self._fun_edge(
                        vertices_coords, i_node, param_dim, Y)
            else:
                if param_dim == 0:
                    is_in_half_plane = Y <= self._fun_edge(
                        vertices_coords, i_node, param_dim, X)
                else:
                    is_in_half_plane = X <= self._fun_edge(
                        vertices_coords, i_node, param_dim, Y)

            if res is None:
                res = is_in_half_plane
            else:
                res = is_in_half_plane*res
        return res

    def _is_in_this_disc(self, disc_id, X, Y):
        center_coords = self.objects[disc_id]["center_coords"]
        radius = self.objects[disc_id]["radius"]
        Xc = center_coords[0]*np.ones_like(X)
        Yc = center_coords[1]*np.ones_like(Y)

        input_coords = np.stack([X, Y], -1)
        aug_center_coords = np.stack([Xc, Yc], -1)
        diff = input_coords-aug_center_coords
        distance_to_center = np.linalg.norm(diff, axis=-1)
        return distance_to_center <= radius

    def is_in_this_object(self, object_id, X, Y):
        '''
        Check whether the given coordinates touch a specific object of the environment or not

        Args:
         - X, Y : np.array, list or float
            the coordinates of the points to probe

        Returns:
         - res : np.array(bool), list[bool] or bool
            True if the given coordinates lead to touch an object
        '''
        assert object_id in self.objects, "Wrong object_id, the object to check is not in the environnement"

        if self.objects[object_id]["type"] == "polygon":
            return self._is_in_this_polygon(object_id, X, Y)
        else:
            assert self.objects[object_id]["type"] == "disc", "Not implemented object type"
            return self._is_in_this_disc(object_id, X, Y)

    def is_in_an_object(self, X, Y):
        '''
        Check whether the given coordinates touch an object of the environment or not

        Args:
         - X, Y : np.array, list or float
            the coordinates of the points to probe

        Returns:
         - res : np.array(bool), list[bool] or bool
            True if the given coordinates lead to touch an object
        '''

        res = None
        for object_id in self.objects.keys():
            is_in_this_object = self.is_in_this_object(object_id, X, Y)
            res = is_in_this_object if res is None else is_in_this_object+res
        return res

    def plot_polygon_boundaries(self, polygon_id):
        """
        Plot the straight lines along the segments of the polygon
        """
        vertices_coords = self.objects[polygon_id]["vertices_coords"]

        X = np.linspace(
            self.env_bounds[0], self.env_bounds[2], self.resolution_plot)
        Y = np.linspace(
            self.env_bounds[1], self.env_bounds[3], self.resolution_plot)
        plt.xlim((self.env_bounds[0], self.env_bounds[2]))
        plt.ylim((self.env_bounds[1], self.env_bounds[3]))

        for i_node in range(vertices_coords.shape[0]):
            param_dim = self._get_param_dim(vertices_coords, i_node)
            if param_dim == 0:
                plt.plot(X, self._fun_edge(vertices_coords,
                         i_node, param_dim, X), label=str(i_node))
            else:
                plt.plot(self._fun_edge(vertices_coords, i_node,
                         param_dim, Y), Y, label=str(i_node))
        plt.legend()
        plt.show()

    # depuis une seule coordonnée ou depuis l'id de l'objet ?
    def remove_objects(self, object_ids: set):
        """
        Remove the objects referred to in object_ids
        """
        new_objects = {}
        np_object_ids = np.array([e for e in object_ids])
        for object_k in self.objects.keys():
            if object_k not in object_ids:
                shift = np.sum(np.where(np_object_ids < object_k, 1, 0))
                new_objects[object_k-shift] = deepcopy(self.objects[object_k])
        self.objects = new_objects

    def _update_map(self):
        map = None
        for object_id in self.objects.keys():
            map_object = self.objects[object_id]["map"]
            map = map_object if map is None else map+map_object
        self.map = map

    def plot_map(self):
        """
        Give a discrete representation of the environment
        """
        X = np.linspace(
            self.env_bounds[0], self.env_bounds[2], self.resolution_plot)
        Y = np.linspace(
            self.env_bounds[1], self.env_bounds[3], self.resolution_plot)

        xv, yv = np.meshgrid(X, Y)
        self._update_map()
        plt.scatter(xv, yv, c=self.map, cmap="winter")
        plt.show()

    def plot_map_object(self, object_id):
        """
        Give a discrete representation of the object alone in the environment
        """
        X = np.linspace(
            self.env_bounds[0], self.env_bounds[2], self.resolution_plot)
        Y = np.linspace(
            self.env_bounds[1], self.env_bounds[3], self.resolution_plot)

        xv, yv = np.meshgrid(X, Y)

        map_object = self.objects[object_id]["map"]
        plt.scatter(xv, yv, c=map_object, cmap="winter")
        plt.show()
