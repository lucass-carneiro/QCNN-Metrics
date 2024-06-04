import quantum_derivatives as qd
import domain_map as dm

class HagenPoiseuille:
    def __init__(self, x0: float, xf: float, G: float, R: float, mu: float):
        self.map = dm.LinearMap(x0, xf)
        self.G = G
        self.R = R
        self.mu = mu
    
    def cost_int_pointwise(self, node, weights, x):
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
        # BCs
        bc_l = (node(weights, x=self.map.global2local(self.map.global_start)) - self.G * self.R**2 / (4.0 * self.mu))**2
        bc_r = (node(weights, x=self.map.global2local(self.map.global_end)))**2
        bc_d = (qd.df(node, weights, x=self.map.global2local(self.map.global_start)))**2

        # Interior cost
        int_cost = sum(self.cost_int_pointwise(node, weights, x) ** 2 for x in data)

        return (bc_l + bc_r + bc_d + int_cost) / N
    
class PlaneHagenPoiseuille:
    def __init__(self, x0: float, xf: float, G: float, R: float, mu: float):
        self.map = dm.LinearMap(x0, xf)
        self.G = G
        self.R = R
        self.mu = mu
    
    def cost_int_pointwise(self, node, weights, x):
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
        # BCs
        bc_l = (node(weights, x=self.map.global2local(self.map.global_start)))**2
        bc_r = (node(weights, x=self.map.global2local(self.map.global_end)))**2

        # Interior cost
        int_cost = sum(self.cost_int_pointwise(node, weights, x) ** 2 for x in data)

        return (bc_l + bc_r + int_cost) / N