# -*- coding: utf-8 -*-

import ufl
import numpy as np

#shape damage function
def w(alpha):
    return alpha*alpha

def w_prime(alpha):
    return 2*alpha #1.0

#Degradation function
def gk(alpha, k= 1e-6):
    return (1-alpha)**2 + k

def gk_prime(alpha, k= 1e-6):
    return 2*(alpha-1)


def epsilon(u):
    return ufl.sym(ufl.nabla_grad(u))

def epsilon_mfront(u):
    e = ufl.sym(ufl.nabla_grad(u))
    return ufl.as_vector([e[0,0], e[1,1], 0., np.sqrt(2)*e[0,1]])

def gradient_transformation(u, ndim):
    f = ufl.grad(u) + ufl.Identity(ndim)
    return ufl.as_vector([f[0,0], f[1,1], 1., f[0,1],f[1,0]])

class ElasticQSModel:
    def __init__(self , ndim, material_parameters):
        mp = material_parameters
        self.ndim = ndim
        self.lmbda = mp["lmbda"]
        self.mu = mp["mu"]

    def sigma_0(self,u):
        return  self.lmbda * ufl.tr( epsilon(u)) *  ufl.Identity(self.ndim)  + 2* self.mu *  epsilon(u)

    def psi_0(self, u):
        eps = epsilon(u)
        return 0.5*self.lmbda* ufl.tr(eps)**2 +  self.mu *ufl.inner(eps,eps)


class BourdinModel:
    def __init__(self , ndim, material_paramters):
        mp = material_paramters
        self.ndim = ndim
        self.lmbda = mp["lmbda"]
        self.mu = mp["mu"]
        self.K0 = mp["K0"]

    def sigma_0(self,u):
        return  self.lmbda * ufl.tr( epsilon(u)) *  ufl.Identity(self.ndim)  + 2* self.mu *  epsilon(u)

    def psi_0(self, u):
        eps = epsilon(u)
        return 0.5*self.lmbda* ufl.tr(eps)**2 +  self.mu *ufl.inner(eps,eps)

    def sigma(self,u, alpha):
        return  gk(alpha, k= 1e-6) * self.sigma_0(u)
    

    def psi(self, u, alpha):
        return gk(alpha, k= 1e-6) * self.psi_0(u)


class AmorModel:
    def __init__(self , ndim, material_paramters):
        mp = material_paramters
        self.ndim = ndim
        self.lmbda = mp["lmbda"]
        self.mu = mp["mu"]
        self.K0 = mp["K0"]

    def sigma_0(self,u):
        return  self.lmbda * ufl.tr( epsilon(u)) *  ufl.Identity(self.ndim)  + 2* self.mu *  epsilon(u)

    def psi_0(self, u):
        eps = epsilon(u)
        return 0.5*self.lmbda* ufl.tr(eps)**2 +  self.mu *ufl.inner(eps,eps)

    def psi_plus(self, u):
        eps = epsilon(u)
        pp  = ufl.conditional(ufl.ge(ufl.tr(eps), 0),ufl.tr(eps),0 )
        return 0.5* self.K0* pp**2 + self.mu *ufl.inner(ufl.dev(eps),ufl.dev(eps))

    def psi_moins(self, u):
        eps = epsilon(u)
        pn = ufl.conditional(ufl.lt(ufl.tr(eps), 0),ufl.tr(eps),0 )
        return 0.5 * self.K0 * pn**2

    def psi_damage(self, u, alpha):
        return  gk(alpha, k= 1e-6)*self.psi_plus(u)  + self.psi_moins(u)

    def sigma(self, u, alpha):
        eps = epsilon(u)
        pp  = ufl.conditional(ufl.ge(ufl.tr(eps), 0),ufl.tr(eps),0 )
        pn = ufl.conditional(ufl.lt(ufl.tr(eps), 0),ufl.tr(eps),0 )
        sig_p  = self.K0 * pp * ufl.Identity(self.ndim) + 2* self.mu *ufl.dev(eps)
        sig_n  = self.K0 * pn * ufl.Identity(self.ndim)
        return  gk(alpha, k= 1e-6) * sig_p + sig_n

