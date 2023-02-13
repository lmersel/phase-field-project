from dolfinx import fem
import ufl

from petsc4py import PETSc
import numpy as np

class NewtonRaphsonSolver:
    def __init__(self, F, u, bc, comm):
        self.u = u
        V_u = u.function_space
        self.u_trial = ufl.TrialFunction(V_u)
        self.L = fem.form(-F)
        self.a = fem.form( ufl.derivative(F, self.u, self.u_trial))
        self.bcu = bc
        self.comm = comm
        #self.bcs_hom = bc_hom

    def residual(self, b):
        """ Assemble residual vector"""
        with b.localForm() as b_local:
            b_local.set(0.0)
        fem.petsc.assemble_vector(b, self.L)
        fem.petsc.apply_lifting(b, [self.a], [self.bcu])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, self.bcu)
        return b

    def jacobian(self,J):
        """Assemble Jacobian matrix"""
        #J.zeroEntries
        fem.petsc.assemble_matrix(J, self.a, self.bcu)
        J.assemble()
        return J
   
    def solve_NR(self, bc_hom, max_iter=100, rtol = 1e-9):

        fem.petsc.set_bc(self.u.vector, self.bcu)
        #print("before hom: u.x.array[:] = ", self.u.x.array[:])
        self.bcu = bc_hom
        
        Jj = fem.petsc.create_matrix(self.a)
        bb = fem.petsc.create_vector(self.L)
        b = self.residual(bb)
        J = self.jacobian(Jj)

        delta_u = J.createVecRight()
        r = 1.0
        k = 0

        while r > rtol and k < max_iter:
            """create lu petcs solver"""
            solverLU = PETSc.KSP().create(self.comm)
            #solverLU.setType('cg')
            solverLU.setType('preonly')
            pc = solverLU.getPC()
            pc.setType('lu')

            """ Set matrix operator"""
            solverLU.setOperators(J)

            """ Compute solution"""
            solverLU.solve(b, delta_u)
            #print("after : delta_u.array[:] = ", delta_u.array[:])

            """ Monitoring"""
            residu = J * delta_u - b
            if b.norm() == 0.0:
                r = 0.0
            else :
                r = residu.norm()/ b.norm()
                
            print("|J * du - b|/|b|", r)
            
            # update the solution
            self.u.x.array[:] = self.u.x.array[:] + delta_u.array[:]
            #print("after : u.x.array[:] = ", u.x.array[:])
            
            b = self.residual(bb)
            J = self.jacobian(Jj)
            k+= 1
