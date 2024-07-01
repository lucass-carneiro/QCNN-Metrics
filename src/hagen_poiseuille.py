"""
A problem type defined by training a quantum circuit to solve the
Hagen-Poiseuille ODE.
"""

import problem as prb
import quantum_derivatives as qd
import domain_map as dm


class HagenPoiseuille(prb.Problem):
    """
    Solves the Hagen-Poiseuille equation, as defined [here](https://en.wikipedia.org/wiki/Hagen%E2%80%93Poiseuille_equation).
    """

    def __init__(self, x0: float, xf: float, G: float, R: float, mu: float):
        """
        Initializes the problem object.

        Parameters:
            x0 (float): Left boundary of the domain.
            xf (float): Righr boundary of the domain.
            G (float): Hagen Poiseuille `G` parameter.
            R (float): Hagen Poiseuille `R` parameter.
            mu (float): Hagen Poiseuille `mu` parameter.
        """
        self.map = dm.LinearMap(x0, xf)
        self.G = G
        self.R = R
        self.mu = mu

    def get_domain_map(self):
        """
        Returns:
         (DomainMap): The linear domain map used in the problem.
        """
        return self.map

    def cost_int_pointwise(self, node, weights, x):
        """
        Returns the cost function of the problem at each point in the
        input domain.

        Parameters:
          node (QuantumNode): The quantum node used for evaluating the circuit.
          weights (array): The array of weights representing the free parameters in the circuit.
          x (float): The point where to comput the cost function.
        """
        # Get local X
        X = self.map.global2local(x)

        # Compute derivatives in local space
        l_dfdX = qd.df(node, weights, x=X)
        l_d2fdX2 = qd.d2f(node, weights, x=X)

        # Compute jacobians
        dldg = self.map.dlocal_dglobal(x)
        d2ldg2 = self.map.d2local_dglobal2(x)

        # Compute derivatives in global space
        g_dfdx = dldg * l_dfdX
        g_d2fdx2 = dldg * dldg * l_d2fdX2 + d2ldg2 * l_dfdX

        # ODE in global space
        return (g_d2fdx2 + self.G/self.mu) * x + g_dfdx

    def cost(self, node, weights, data, N):
        """
        Returns the cost function of the problem.

        Parameters:
          node (QuantumNode): The quantum node used for evaluating the circuit.
          weights (array): The array of weights representing the free parameters in the circuit.
          data (array): The input domain data of the problem.
          N (int): The size of the input domain data.
        """
        # BCs
        bc_l = (node(weights, x=self.map.global2local(
            self.map.global_start)) - self.G * self.R**2 / (4.0 * self.mu))**2
        bc_r = (node(weights, x=self.map.global2local(self.map.global_end)))**2
        bc_d = (qd.df(node, weights, x=self.map.global2local(
            self.map.global_start)))**2

        # Interior cost
        int_cost = sum(self.cost_int_pointwise(
            node, weights, x) ** 2 for x in data)

        return (bc_l + bc_r + bc_d + int_cost) / N


class PlaneHagenPoiseuille(prb.Problem):
    """
    Solves the Hagen-Poiseuille equation between to infinite plates as defined [here](https://en.wikipedia.org/wiki/Hagen%E2%80%93Poiseuille_equation).
    """

    def __init__(self, x0: float, xf: float, x, G: float, R: float, mu: float):
        """
        Initializes the problem object.

        Parameters:
            x0 (float): Left boundary of the domain.
            xf (float): Righr boundary of the domain.
            G (float): Hagen Poiseuille `G` parameter.
            R (float): Hagen Poiseuille `R` parameter.
            mu (float): Hagen Poiseuille `mu` parameter.
        """
        self.map = dm.LinearMap(x0, xf)
        self.G = G
        self.R = R
        self.mu = mu
        self.target = self.G / (2 * self.mu) * x * (xf - x)

    def get_domain_map(self):
        """
        Returns:
         (DomainMap): The linear domain map used in the problem.
        """
        return self.map

    def cost_int_pointwise(self, node, weights, x):
        """
        Returns the cost function of the problem at each point in the
        input domain.

        Parameters:
          node (QuantumNode): The quantum node used for evaluating the circuit.
          weights (array): The array of weights representing the free parameters in the circuit.
          x (float): The point where to comput the cost function.
        """
        # Get local X
        X = self.map.global2local(x)

        # Compute derivatives in local space
        l_dfdX = qd.df(node, weights, x=X)
        l_d2fdX2 = qd.d2f(node, weights, x=X)

        # Compute jacobians
        dldg = self.map.dlocal_dglobal(x)
        d2ldg2 = self.map.d2local_dglobal2(x)

        # Compute derivatives in global space
        g_d2fdx2 = dldg * dldg * l_d2fdX2 + d2ldg2 * l_dfdX

        # ODE in global space
        return g_d2fdx2 + self.G/self.mu

    def cost(self, node, weights, data, N):
        """
        Returns the cost function of the problem.

        Parameters:
          node (QuantumNode): The quantum node used for evaluating the circuit.
          weights (array): The array of weights representing the free parameters in the circuit.
          data (array): The input domain data of the problem.
          N (int): The size of the input domain data.
        """
        # BCs
        bc_l = (node(weights, x=self.map.global2local(self.map.global_start)))**2
        bc_r = (node(weights, x=self.map.global2local(self.map.global_end)))**2

        # Interior cost
        int_cost = sum(self.cost_int_pointwise(
            node, weights, x) ** 2 for x in data)

        return (bc_l + bc_r + int_cost) / N
