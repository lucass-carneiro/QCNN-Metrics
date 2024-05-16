class DomainMap:
    def __init__(self, global_start: float, global_end: float):
        self.global_start = global_start
        self.global_end = global_end

    def local2global(self, local_X):
        return local_X
    
    def global2local(self, global_x):
        return global_x
    
    def dlocal_dglobal(self, global_x):
        # Obtained by computing the derivative of global2local
        return 0.0
    
    def d2local_dglobal2(self, global_x):
        # Obtained by computing the second derivative of global2local
        return 0.0
    
class LinearMap(DomainMap):
    def local2global(self, local_X):
        return (self.global_start + self.global_end + (self.global_end - self.global_start) * local_X) / 2.0
    
    def global2local(self, global_x):
        return (self.global_start + self.global_end - 2.0 * global_x) / (self.global_start - self.global_end)

    def dlocal_dglobal(self, _):
        # Obtained by computing the derivative of global2local
        return 2.0 / (self.global_end - self.global_start)


    def d2local_dglobal2(self, _):
        # Obtained by computing the second derivative of global2local
        return 0.0