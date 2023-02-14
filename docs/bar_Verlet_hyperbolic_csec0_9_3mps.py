from dolfinx import log, io, mesh, fem, cpp, plot
import ufl

from petsc4py import PETSc
from mpi4py import MPI

import numpy as np

import material_model as bv
import newton_raph_solver as nrs

import util_func as ut

import mgis.behaviour as mgis_bv

# interpolate() can have an arg typing callable, FUnction() ou Expression() 
class Load():
    v0 = 3.0 #m/s
    t0 = 0.00001 #s
    def __init__(self, ndim, t):
        self.t = t
        self.ndim = ndim

    def __call__(self,x):
        """ need a method __call__ to build a load evolving in time
        x : an array of coordinates related to the SpaceFunction use to define the the fem.Function(SpaceV_u)
        ==> fem.Function(SpaceV_u).interpolate(object Load(t))
        """
        values = np.zeros((self.ndim-1, x.shape[1]),dtype=PETSc.ScalarType)
        if self.t < Load.t0:
            values[0] = 0.5 * (Load.v0 / Load.t0) * self.t * self.t
        else:
            values[0] = Load.v0 * (self.t-Load.t0) + 0.5 * Load.v0 * Load.t0
        return values

class FractureProblem:
    def __init__(self, msh, behavior, material_parameters, savedir):
        self.damage = 'explicit02' # 'implicit', "explicit02"
        self.meca = 'explicit'
        self.staggered_solver = dict({"iter_max":1,"tol":1e-4})
        self.time_parameter = dict({"t_max":0.0005,"Nsteps":1000})
        self.savedir = savedir
        mp = material_parameters
        self.msh = msh
        self.mat = behavior

        self.dynamic_meca = True
        self.dynamic_damage = True
        self.explicit = True

        if self.dynamic_damage :
             self.eta  = mp["viscosity"]
             self.rho_d = mp["rho_damage"]
        if self.dynamic_meca:
            self.rho = mp["rho"]
            
        self.set_functions()
        
        self.dx = ufl.Measure("dx")
        self.ds = ufl.Measure("ds")
        self.comm = MPI.COMM_WORLD
        self.bcu = []
        self.bcu_hom = []

        self.cw = mp["cw"]
        self.lc = mp["lc"]
        self.Gc = mp["Gc"]
        self.E  = mp["E"]
        self.nu  = mp["nu"]
        self.phi_c = mp["threshold"]
        self.celerity = mp["celerity"]

        self.boundary_indices = {"left": 0, "right": 1, "top": 2, "bottom": 3, "impact": 4}
        self.Ly = self.msh.geometry.x[:,1].max()
        self.Lx = self.msh.geometry.x[:,0].max()
        self.ndim = self.msh.topology.dim
        
        # notice that cpp.mesh.h(self.msh, self.ndim, [0])[0] gives the diagonal size of the square cell
        # time stepping
        self.h_cell = cpp.mesh.h(self.msh, self.ndim, [0])[0]
        self.set_time_stepping()
        # for postprocessing
        self.loading  = []
        self.reaction = []
        self.energie  = []


        # Export xdmf flag
        self.save_stress = True
        self.save_strain = True #False

    def mpi_print(self, s):
        print(f"Rank {self.comm.rank}: {s}")

    def set_time_stepping(self):
        """ define time stepping in terms of the regime"""
        self.dt_CFL = self.h_cell/np.sqrt(self.E/self.rho)
        if self.comm.rank ==0:
            print("Pas de temps critique en meca {:.2e}s".format(self.dt_CFL))
        self.c_sec = 0.9

        if self.meca == "implicit" and self.damage =="implicit":
            self.increment = np.linspace(0, self.time_parameter["t_max"], self.time_parameter["Nsteps"])
            self.dt = self.time_parameter["t_max"]/self.time_parameter["Nsteps"]
            if self.comm.rank ==0:
                print("(1) Pas de temps constant utilisé {:.2e}s".format(self.dt))
        if self.meca == "explicit" and self.damage =="implicit":
            self.increment = np.arange(0, self.time_parameter["t_max"], self.c_sec *self.dt_CFL)
            self.dt = self.c_sec * self.dt_CFL
            if self.comm.rank ==0:
                print("(2) Pas de temps constant utilisé {:.2e}s".format(self.dt))
        if self.meca == "explicit" and self.damage == "explicit01":
            self.dt_CFL = self.h_cell/np.sqrt(self.E/self.rho)
            self.dtc_d = (self.cw * self.h_cell * self.h_cell * self.eta)/ (2*self.lc*self.Gc)
            self.dt = self.c_sec * min(self.dt_CFL, self.dtc_d)
            self.increment = np.arange(0, self.time_parameter["t_max"], self.dt)
            if self.comm.rank ==0:
                print("Pas de temps critique en endo {:.2e}s".format(self.dtc_d))
                print("(3) Pas de temps constant utilisé {:.2e}s".format(self.dt))
        if self.meca == "explicit" and self.damage == "explicit02":
            self.dt_CFL = self.h_cell/np.sqrt(self.E/self.rho)
            #self.dtc_d = (self.cw * self.h_cell * self.eta)/ (2*self.lc*self.Gc)
            print("Pas de temps critique en endo MODIFIE --- dtc_d = 2*h*lc/c")
            self.dtc_d = 2*(self.h_cell * self.lc)/ (self.celerity)
            self.dt = self.c_sec * min(self.dt_CFL, self.dtc_d)
            self.increment = np.arange(0, self.time_parameter["t_max"], self.dt)
            if self.comm.rank ==0:
                print("Pas de temps critique en endo {:.2e}s".format(self.dtc_d))
                print("(4) Pas de temps constant utilisé {:.2e}s".format(self.dt))
            #self.mpi_print(f"dtc : {self.dt}")
            
        
    def set_functions(self):
        
        self.degS, degH = 2 , 2
        self.dim_F, self.dim_S = 5, 4
        
        FE_vector = ufl.VectorElement("CG", self.msh.ufl_cell(),1)
        self.V_u       = fem.FunctionSpace(self.msh, FE_vector)

        FE_scalar=ufl.FiniteElement("CG",msh.ufl_cell(),1)
        self.V_alpha=fem.FunctionSpace(msh,FE_scalar)
        
        self.V_H       = fem.FunctionSpace(self.msh,('DG', 0))
        self.Tens      = fem.TensorFunctionSpace(self.msh, ('DG', 0))

        # for the postprocessing of history varible on dg space
        element_quad_H = ufl.FiniteElement("Quadrature",self.msh.ufl_cell(), degH, quad_scheme='default') 
        self.V_H_quad       = fem.FunctionSpace(msh,element_quad_H )

        FE_quad_vec = ufl.VectorElement("Quadrature",self.msh.ufl_cell(),self.degS, dim = self.dim_F ,quad_scheme='default')
        self.V_F       = fem.FunctionSpace(self.msh, FE_quad_vec)

        #element_quad_stress = ufl.TensorElement("Quadrature", msh.ufl_cell(), degS, shape = ((2,2)), quad_scheme ='default')
        element_quad_stress = ufl.VectorElement("Quadrature", self.msh.ufl_cell(), self.degS, dim = self.dim_S, quad_scheme ='default')
        self.V_tens   = fem.FunctionSpace(self.msh,element_quad_stress)

        # for the postprocessing of stress and strain on dg space
        self.FE_vector_dg = ufl.VectorElement("DG",self.msh.ufl_cell(), degree = 0, dim = self.dim_S ,quad_scheme='default')
        self.V_dg       = fem.FunctionSpace(self.msh,self.FE_vector_dg)
        

        # function definition
        self.u_trial     =  ufl.TrialFunction(self.V_u)
        self.u_test      =  ufl.TestFunction(self.V_u)
        self.u           = fem.Function(self.V_u, name = 'Displacement')
        self.uu           = fem.Function(self.V_u, name = 'Disp_proj')
        self.u_old       = fem.Function(self.V_u, name = 'previous_iteration_displacement')
        self.alpha_trial = ufl.TrialFunction(self.V_alpha)
        self.alpha_test        = ufl.TestFunction(self.V_alpha)
        self.alpha       = fem.Function(self.V_alpha, name = 'Damage')
        self.alpha_old   = fem.Function(self.V_alpha, name = 'previous_iteration_damage')

        self.H_new       = fem.Function(self.V_H, name = "current_iteration_history")
        self.H_old       = fem.Function(self.V_H, name ="previous_iteration_history")
        self.H_inc       = fem.Function(self.V_H, name ="previous_increment_history")
        self.sig_funct   = fem.Function(self.Tens, name = "stress") #dg
        self.eps_funct   = fem.Function(self.Tens, name = "strain") #dg

        self.psi_funct  = fem.Function(self.V_H, name ="elastic_energy_density" )

        if self.dynamic_meca:
            self.u_p    = fem.Function(self.V_u, name ='Velocity') #  velocity
            self.u_pp   = fem.Function(self.V_u, name ='Acceleration')  # acceleration
            self.u_pp_old    = fem.Function(self.V_u, name ='Previous Acceleration') # acceleration
        if self.dynamic_damage:
            self.alpha_p = fem.Function(self.V_alpha, name ='Damage_velocity')
            self.alpha_p_old = fem.Function(self.V_alpha, name ='Previous_Damage_velocity') 
            self.alpha_pp_old = fem.Function(self.V_alpha, name ='Previous_Damage_acceleration') 
            self.alpha_pp = fem.Function(self.V_alpha, name ='Damage_acc') #  velocity
            self.alpha_pred = fem.Function(self.V_alpha, name ='_Damage_velocity_predictor')
            self.alpha_p_pred = fem.Function(self.V_alpha, name ='Damage_velocity_predictor')
            self.M_control      = fem.Function(self.V_H, name = "mobility_control")
            


        # Env mfront
        #________________________________________________________
        self.sig_quad   = fem.Function(self.V_tens, name = "stress_quad")
        self.Egreen_quad   = fem.Function(self.V_tens, name = "strain_GrLg_quad")
        self.eps_quad   = fem.Function(self.V_tens, name = "strain_quad")
        
        self.alpha_mfront = fem.Function(self.V_H_quad, name = "Damage_projected_on_Qspace")
        #self.H_new_dg     = fem.Function(self.V_H, name="current_iteration_history_dg")
        self.H_new_quad       = fem.Function(self.V_H_quad, name = "current_iteration_history")

        self.sig_dg = fem.Function(self.V_dg, name="stress_dg")
        self.eps_dg = fem.Function(self.V_dg, name="strain_dg")

        self.F_grad = fem.Function(self.V_F, name = "transformation_gradient")
 
        #________________________________________________________
        

    def set_initilization(self):
        """ Initialization """
        
        # initialize the unknown displacement and damage solution
        def u_zeros_init(x):
            values = np.zeros((2, x.shape[1]))
            values[0] = 0.0
            values[1] = 0.0
            return values
        self.u_old.interpolate(u_zeros_init)

        def alpha_zeros_init(x):
            values = np.zeros((1, x.shape[1]))
            return values
        self.alpha_old.interpolate(alpha_zeros_init)

        # initialize the load
        self.load_func = fem.Function(self.V_u.sub(0).collapse()[0])
        self.impact_load = Load(ndim,t=0.)

        # initialize the driving force (history H)
        #self.psi_pos_expr = fem.Expression(self.mat.psi_0(self.u), self.V_H.element.interpolation_points())
        #self.H_new.interpolate(self.psi_pos_expr) # Initialisation

        # initialize XDMFfile file
        with io.XDMFFile(self.comm, self.savedir+"displacement.xdmf", "w") as self.xdmf0:
            self.xdmf0.write_mesh(self.msh)
        with io.XDMFFile(self.comm, self.savedir+"damaga.xdmf", "w") as self.xdmf1:
            self.xdmf1.write_mesh(self.msh)
        with io.XDMFFile(self.comm, self.savedir+ "velocity.xdmf", "w") as self.xdmf4:
            self.xdmf4.write_mesh(self.msh)
        with io.XDMFFile(self.comm, self.savedir+ "acceleration.xdmf", "w") as self.xdmf5:
            self.xdmf5.write_mesh(self.msh)

        if self.save_stress:
            with io.XDMFFile(self.comm, self.savedir+"stress.xdmf", "w") as self.xdmf2:
                self.xdmf2.write_mesh(self.msh)

        if self.save_strain:
            with io.XDMFFile(self.comm, self.savedir+"strain.xdmf", "w") as self.xdmf3:
                self.xdmf3.write_mesh(self.msh)
                
        # initialize projection en Quadrature space for : transfo grad, F and phase-field, alpha
        
        ut.local_project(self.msh, self.alpha, self.V_H_quad, self.alpha_mfront)

        ut.local_project(self.msh, self.Egreen_quad, self.V_dg, self.eps_dg)
        ut.local_project(self.msh, self.sig_quad, self.V_dg, self.sig_dg)
        ut.local_project(self.msh, self.H_new_quad, self.V_H, self.H_new)
        
        # initialize MFront/MGIS environnement

        # Defining the modelling hypothesis
        self.h = mgis_bv.Hypothesis.PlaneStrain

        # # finite strain options
        # self.bopts = mgis_bv.FiniteStrainBehaviourOptions()
        # self.bopts.stress_measure = mgis_bv.FiniteStrainBehaviourOptionsStressMeasure.PK2 
        # self.bopts.tangent_operator = mgis_bv.FiniteStrainBehaviourOptionsTangentOperator.DS_DEGL 
        # Loading the behaviour
        lib = 'src/libBehaviour.so'
        behaviour_label = "PhaseFieldDisplacementNguyenSplit"
        #self.bb = mgis_bv.load(self.bopts,lib ,behaviour_label ,self.h)

        # small def : no need of bopts
        self.bb = mgis_bv.load(lib ,behaviour_label ,self.h)
        ncells = self.msh.topology.index_map(self.ndim).size_local
        ncells0 = self.msh.topology.index_map(self.ndim).num_ghosts
        print('ncell ghost :', ncells0)
        self.ngauss = (ncells+ncells0) * 4 #it's a supposition of nm of integration points
        #self.ngauss = (ncells) * 4
        self.m = mgis_bv.MaterialDataManager(self.bb, self.ngauss)


        for s in [self.m.s0, self.m.s1]:
            mgis_bv.setMaterialProperty(s, "YoungModulus", self.E)
            mgis_bv.setMaterialProperty(s, "PoissonRatio", self.nu)
            mgis_bv.setExternalStateVariable(s, 'Temperature', 293.15)
            mgis_bv.setExternalStateVariable(s, 'Damage', 0.0)


        self.u.x.array[:] = self.u.x.array[:] + self.dt * self.u_p.x.array[:] + 0.5 * self.dt**2 * self.u_pp_old.x.array[:]
        fem.set_bc(self.u.x.array, self.bcu)

        eps_expr = fem.Expression(bv.epsilon_mfront(self.u), self.V_dg.element.interpolation_points())
        self.eps_dg.interpolate(eps_expr)
        #print('eps_fenicsx : ',self.eps_dg.x.array[:] )

        # METHOD 2 : Small Deformation
        # strain doesn't belong to internal_state_variable anymore
        
        ut.local_project(self.msh, bv.epsilon_mfront(self.u), self.V_tens, self.eps_quad)
        self.m.s0.gradients[:,:] = self.eps_quad.x.array.reshape(self.m.n, self.dim_S)
        self.m.s1.gradients[:,:] = self.eps_quad.x.array.reshape(self.m.n, self.dim_S)
        #print('eps_mfront (s0) : ',self.m.s0.gradients[:,:] )

        mgis_bv.integrate(self.m,mgis_bv.IntegrationType.IntegrationWithoutTangentOperator , 0, 0, self.m.n)



    def meshtag(self):
        """ marker entities"""
        def right(x):
            return np.isclose(x[0], self.Lx)
        def left(x):
            return np.isclose(x[0], 0)
        def bottom(x):
            return np.isclose(x[1], 0)
        def top(x):
            return np.isclose(x[1], self.Ly)


        self.facet_left_dofs = fem.locate_dofs_geometrical((self.V_u.sub(0), self.V_u.sub(0).collapse()[0]), left)
        self.facet_right_dofs = fem.locate_dofs_geometrical((self.V_u.sub(0), self.V_u.sub(0).collapse()[0]), right)
        self.facet_bottom_dofs = fem.locate_dofs_geometrical((self.V_u.sub(1), self.V_u.sub(1).collapse()[0]), bottom)
        
        facet_top     = mesh.locate_entities(self.msh, self.ndim-1, top)
        facet_right   = mesh.locate_entities(self.msh, self.ndim-1, right)
        facet_left    = mesh.locate_entities(self.msh, self.ndim-1, left)
        facet_bottom  = mesh.locate_entities(self.msh, self.ndim-1, bottom)


        self.facet_left_dofs_alpha = fem.locate_dofs_topological(self.V_alpha, self.ndim-1, facet_left)
        self.facet_right_dofs_alpha = fem.locate_dofs_topological(self.V_alpha, self.ndim-1, facet_right)

        self.facet_indices, self.facet_markers = [], []
        self.facet_indices.append(facet_top)
        self.facet_indices.append(facet_bottom)
        self.facet_indices.append(facet_right)
        self.facet_indices.append(facet_left)
        

        self.facet_markers.append(np.full(len(facet_top ),  self.boundary_indices["top"]))
        self.facet_markers.append(np.full(len(facet_bottom ), self.boundary_indices["bottom"]))
        self.facet_markers.append(np.full(len(facet_right ), self.boundary_indices["right"]))
        self.facet_markers.append(np.full(len(facet_left ), self.boundary_indices["left"]))
        
        

        # create MeshTags identifying the facets for each boundary condition
        self.facet_indices = np.array(np.hstack(self.facet_indices), dtype=np.int32)
        self.facet_markers = np.array(np.hstack(self.facet_markers), dtype=np.int32)
        sorted_facets = np.argsort(self.facet_indices)
        self.facet_tag = mesh.meshtags(self.msh, self.ndim-1, self.facet_indices[sorted_facets], self.facet_markers[sorted_facets])


        
    def set_variational_formulation(self):
        
        self.meshtag()
        
        metadata = {"quadrature_degree": 4}
        self.ds = ufl.Measure("ds", domain=self.msh, subdomain_data=self.facet_tag, metadata=metadata)
        self.dx = ufl.Measure("dx", domain=self.msh,  metadata=metadata)

        metadata_quad = {"quadrature_degree": self.degS, "quadrature_scheme": "default"}
        self.dxm = ufl.dx(metadata=metadata_quad)

        # coeff de Newmark for explicit
        self.beta = PETSc.ScalarType(0.25)
        self.gamma = PETSc.ScalarType(0.)
        #self.H_new_quad.x.array[:] = self.m.s1.internal_state_variables[:,0]
        #ut.local_project(self.msh, self.H_new_quad, self.V_H, self.H_new)
        self.M_control.x.array[:] = 0.5 * self.celerity / (2*np.sqrt(4 * self.Gc * self.lc * self.H_new.x.array[:] + self.Gc * self.Gc ))

        # The predictor for damage and damage velocity with beta = 0 and gamma = 1/2
        self.alpha_pred.x.array[:] = self.alpha_old.x.array[:] + self.dt * self.alpha_p_old.x.array[:] + 0.5 * self.dt**2 * self.alpha_pp_old.x.array[:]
        self.alpha_p_pred.x.array[:] = self.alpha_p_old.x.array[:] + self.dt*0.5 * self.alpha_pp_old.x.array[:]

        # self.alpha_pred.x.array[:] = self.alpha_old.x.array[:] + self.dt * self.alpha_p_old.x.array[:] + 0.5 * self.dt**2 * (1-2*self.beta)*self.alpha_pp_old.x.array[:]
        # self.alpha_p_pred.x.array[:] = self.alpha_p_old.x.array[:] + self.dt*(1-self.gamma) * self.alpha_pp_old.x.array[:]

        # Mechanical pb : Mass matrix and lumping
        self.mu_form  = fem.form(self.rho * ufl.inner(self.u_trial, self.u_test) * self.dx)
        M = fem.petsc.assemble_matrix(self.mu_form)
        M.assemble()
        # Matrix Lumping
        # Compute the sum of the rows of the mass matrix
        ones       = fem.Function(self.V_u)
        with ones.vector.localForm() as loc:
           loc.set(1.0)
        self.m_action = ufl.action(self.rho * ufl.inner(self.u_trial, self.u_test) * self.dx, ones)
        
        #print('size mlump : ',m_lumped_diag.size, comm.rank())
        self.m_lumped_diag = fem.petsc.assemble_vector(fem.form(self.m_action))
        #self.m_lumped_diag.ghostUpdate()
        #print('size mlump : ',m_lumped_diag.size,comm.rank())
        
        self.m_lumped_diag_inv = 1./self.m_lumped_diag
        #print('m_action_vec.array:', m_action_vec.array)

        #right
        #M_lumped = np.eye(M.getSize()[0],M.getSize()[0]) * self.m_action_vec.array
        #self.M_lumped_inv = np.linalg.inv(M_lumped)

        
        # Dampage pb : mass and damped matrix and lumping
        # damping_damage matrix
        self.ca_form  = fem.form(self.eta * ufl.inner(self.alpha_trial, self.alpha_test) * self.dx)
        ones_ca       = fem.Function(self.V_alpha)
        with ones_ca.vector.localForm() as loc:
           loc.set(1.0)
        self.ca_action = ufl.action(1.0/self.M_control * ufl.inner(self.alpha_trial, self.alpha_test) * self.dx, ones_ca)
        self.ca_lumped_diag = fem.petsc.assemble_vector(fem.form(self.ca_action))
        #self.ca_lumped_diag_inv = 1./self.ca_lumped_diag

        # mass_damage matrix
        self.ma_form  = fem.form(self.rho_d  * ufl.inner(self.alpha_trial, self.alpha_test) * self.dx)
        ones_ma       = fem.Function(self.V_alpha)
        with ones_ma.vector.localForm() as loc:
           loc.set(1.0)
        self.ma_action = ufl.action(2*self.Gc*self.lc/self.celerity**2 * ufl.inner(self.alpha_trial, self.alpha_test) * self.dx, ones_ma)
        self.ma_lumped_diag = fem.petsc.assemble_vector(fem.form(self.ma_action))
        self.ma_lumped_diag_inv = 1./self.ma_lumped_diag
        
    def define_bc(self):
        
        # clamped in y direction at the bottom edge and x direction at the left edge
        self.zero_u_bottom = fem.Function(self.V_u.sub(1).collapse()[0])
        with self.zero_u_bottom.vector.localForm() as bc_local:
            bc_local.set(0.0)
        self.zero_u_left = fem.Function(self.V_u.sub(0).collapse()[0])
        with self.zero_u_left.vector.localForm() as bc_local:
            bc_local.set(0.0)

        # BUild BC displacement
        self.bc_left = fem.dirichletbc(self.zero_u_left, self.facet_left_dofs , self.V_u.sub(0))
        self.bc_right = fem.dirichletbc(self.load_func , self.facet_right_dofs, self.V_u.sub(0))
        self.bc_bottom = fem.dirichletbc(self.zero_u_bottom,self.facet_bottom_dofs, self.V_u.sub(1))
        self.bcu = [self.bc_right, self.bc_bottom, self.bc_left]

        # BUild BC damage
        self.bcalpha_left = fem.dirichletbc(fem.Constant(self.msh, PETSc.ScalarType(0.)), self.facet_left_dofs_alpha, self.V_alpha)
        self.bcalpha_right = fem.dirichletbc(fem.Constant(self.msh, PETSc.ScalarType(0.)),  self.facet_right_dofs_alpha, self.V_alpha)
        self.bcalpha = [self.bcalpha_right] #[self.bcalpha_right, self.bcalpha_left]
        #self.bcalpha = []

    def define_bc_homogenize(self):
        #Build homogeneous bc for displacement
 #       self.load_func_h = fem.Function(self.V_u.sub(0).collapse()[0])
 #       self.load_0 = Load(ndim, t=0.)
 #       self.load_func_h.interpolate(self.load_0)
        
        
 #       self.bc_right_h = fem.dirichletbc(self.load_func_h, self.facet_right_dofs, self.V_u.sub(0))
 #       self.bcu_hom = [ self.bc_right_h, self.bc_bottom, self.bc_left]


        #Build homogeneous bc for damage
        self.bcalpha_hom = self.bcalpha
        

    def solve_explicit(self):
        self.set_initilization()
        self.set_variational_formulation()

        it = 0
        t_old = 0.
        for t in self.increment[:50]:
            self.impact_load.t = t
            self.load_func.interpolate(self.impact_load)
            if self.comm.rank == 0 :
                print("\n================================================================")
                print("\n Nb increment: %d | Time: %.3g s, Ux_given: %.3g mm  "%(it, t, self.impact_load.t))
                print("\n================================================================")
            #print(self.load_func.x.array)

            self.define_bc()
            it_staggered = 0
            res_H = 1.
            while (res_H > self.staggered_solver["tol"]) and (it_staggered < 1):
                if self.comm.rank ==0:
                    print("self.dt : ", self.dt)
                it_staggered +=1
                
                #----Step 1 : Displacement resolution
                if self.comm.rank ==0:
                    print("\n STEP 1 : Explicit - displacement ")
                    print("_____________________________________")

                self.u.x.array[:] = self.u.x.array[:] + self.dt * self.u_p.x.array[:] + 0.5 * self.dt**2 * self.u_pp_old.x.array[:]
                fem.set_bc(self.u.x.array, self.bcu)
                #print('u.x.array[:] : ',self.u.x.array[:] )

                # -----------------------------
                # Integration mfront
                #-----------------------------
                ut.local_project(self.msh, bv.epsilon_mfront(self.u), self.V_tens, self.eps_quad)
                self.m.s1.gradients[:,:] = self.eps_quad.x.array.reshape((self.m.n,self.dim_S))
                #print('eps_mfront (s1) : ',self.m.s1.gradients[:,:] )

                # check strain from fenicsx project on DG-space
                eps_expr = fem.Expression(bv.epsilon_mfront(self.u), self.V_dg.element.interpolation_points())
                self.eps_dg.interpolate(eps_expr)
                #print('eps_fenicsx : ',self.eps_dg.x.array[:] )

                mgis_bv.integrate(self.m, mgis_bv.IntegrationType.IntegrationWithoutTangentOperator, 0 , 0, self.m.n)#mgis_bv.integrate(m, it, dt , 0, m.n)

                #stress
                self.sig_quad.x.array[:]= self.m.s1.thermodynamic_forces.flatten()
                #print('\nsig from mfront', self.sig_quad.x.array[:])

                # ---------------------------------------
                # Internal force computation
                #----------------------------------------
                f_int_form = fem.form(-ufl.inner( self.sig_quad, bv.epsilon_mfront(self.u_test) ) * self.dxm)
                f_int_vec = fem.petsc.create_vector(f_int_form)
                with f_int_vec.localForm() as f_local:
                    f_local.set(0.0)
                fem.petsc.assemble_vector(f_int_vec, f_int_form)
#                f_int_vec = fem.petsc.assemble_vector( f_int_form)
                
                fem.petsc.apply_lifting(f_int_vec, [self.mu_form], [self.bcu])
                #f_int_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
                fem.petsc.set_bc(f_int_vec, self.bcu)
                
                # ---------------------------------------
                # External force computation
                #----------------------------------------
                f_ext_form = fem.form(ufl.inner(fem.Function(self.V_u), ufl.TestFunction(self.V_u)) * self.ds)
                f_ext_vec = fem.petsc.create_vector(f_ext_form)
                with f_ext_vec.localForm() as f_local:
                    f_local.set(0.0)
                fem.petsc.assemble_vector(f_ext_vec, f_ext_form)
                #fem.petsc.apply_lifting(f_ext_vec, [self.mu_form], [self.bcu])
                fem.petsc.set_bc(f_ext_vec, self.bcu)
                

                # ---------------------------------------
                # Resolution in acceleration
                # a = M_inv_diag* ( Fext - Fint) ; using lumped mass
                #----------------------------------------
                
                # self.u_pp.vector.setArray(self.m_lumped_diag_inv*(f_int_vec+f_ext_vec))
                self.u_pp.vector.array = self.m_lumped_diag_inv*(f_int_vec+f_ext_vec)
                
                # update acceleration and velocity
                self.u_pp.x.scatter_forward()
                self.u_p.x.array[:] = self.u_p.x.array[:] + 0.5*self.dt * (self.u_pp.x.array[:] + self.u_pp_old.x.array[:])
                
                                
                    
                #----Step 2 : Computation of H energy history
                if self.comm.rank == 0:
                    print("\n STEP 2 : Compute history ")
                    print("_____________________________________")

                self.H_new_quad.x.array[:] = self.m.s1.internal_state_variables[:,0]
                ut.local_project(self.msh, self.H_new_quad, self.V_H, self.H_new)
                if self.comm.rank == 0 :
                    self.mpi_print(f'H_new max : {self.H_new.x.array.max()}' )
                #print('H_new : ',self.H_new.x.array[:] )

                #----Step 3 : Damage resolution

                if self.comm.rank ==0:
                    print("\n STEP 3 : Explicit CD/Verlet - damage ")
                    print("_____________________________________")
                """ resolution in acceleration without damping"""
                
                self.M_control.x.array[:] = 0.5 * self.celerity / (2*np.sqrt(4 * self.Gc * self.lc * self.H_new.x.array[:] + self.Gc * self.Gc ))

                self.alpha_pred.x.array[:] = self.alpha_old.x.array[:] + self.dt * self.alpha_p_old.x.array[:] +\
                                             0.5 * self.dt**2 *self.alpha_pp_old.x.array[:]
                self.alpha_p_pred.x.array[:] = self.alpha_p_old.x.array[:] + self.dt*0.5* self.alpha_pp_old.x.array[:]
                
                
                f_int_alpha = fem.form(self.H_new * ufl.inner(self.alpha_test, bv.gk_prime(self.alpha_pred)) * self.dx + \
                                       (self.Gc/(self.cw*self.lc)) * ufl.inner(self.alpha_test, bv.w_prime(self.alpha_pred)) * self.dx +  \
                                       (self.Gc*self.lc)*ufl.inner(ufl.grad(self.alpha_test), ufl.grad(self.alpha_pred))* self.dx)
                b_alpha = fem.petsc.create_vector(f_int_alpha)
                with b_alpha.localForm() as loc:
                    loc.set(0)
                fem.petsc.assemble_vector(b_alpha, f_int_alpha)
                fem.petsc.apply_lifting(b_alpha, [self.ma_form], [self.bcalpha])
                b_alpha.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
                #print('after b_alpha : ', b_alpha.array[:])
                if self.comm.rank ==0:
                    print('\nb_alpha max: ', b_alpha.array.max())
                    print('\nb_alpha min: ', b_alpha.array.min())
                fem.petsc.set_bc(b_alpha, self.bcalpha)

                #self.alpha_pp.vector.getArray(self.ma_lumped_diag_inv * (- b_alpha))
                self.alpha_pp.vector.array = self.ma_lumped_diag_inv * (- b_alpha)
                #self.alpha_pp.x.array[:] = self.ma_lumped_diag_inv * (self.ca_lumped_diag.array * self.alpha_p.x.array - b_alpha.array)


                # compute damage and damage_rate
                self.alpha.x.array[:] = self.alpha_pred.x.array[:] # + self.dt * self.dt * self.beta * self.alpha_pp.x.array[:]
                self.alpha_p.x.array[:] = self.alpha_p_pred.x.array[:] +  0.5 * self.dt * self.alpha_pp.x.array[:]

                # ------------------------------------------------------------------
                # alpha_cg to alpha_quad for mfront : damage as an external variable
                #-------------------------------------------------------------------
                ut.local_project(self.msh, self.alpha, self.V_H_quad, self.alpha_mfront)
                mgis_bv.setExternalStateVariable(self.m.s1, 'Damage',  self.alpha_mfront.x.array[:], mgis_bv.MaterialStateManagerStorageMode.LocalStorage )
                
                #----Step 4 : Residual
                if self.comm.rank == 0 :
                    print("\n STEP 4 : Compute Residual ")
                    print("_____________________________________")

                H_gap = self.H_new.x.array - self.H_old.x.array
                index_max,H_gap_max = np.argmax(H_gap), H_gap.max()
                res_H = abs(H_gap_max)/ self.H_old.x.array[index_max]
                if self.comm.rank == 0 :
                    print('\nMax(|H_n - H_o|) :', H_gap.max())
                    print('\nMax(|H_n - H_o|)/Max(|H_o|) : %g'%(res_H))


                #---- Step 5 : Update previous solution
                self.u_old.x.array[:]        = self.u.x.array[:]
                self.u_pp_old.x.array[:]     = self.u_pp.x.array[:]
                
                self.alpha_old.x.array[:]    = self.alpha.x.array[:]
                self.alpha_p_old.x.array[:]  = self.alpha_p.x.array[:]
                self.alpha_pp_old.x.array[:] = self.alpha_pp.x.array[:]
                self.H_old.x.array[:]        = self.H_new.x.array[:]
                

            if self.comm.rank == 0 :
                print("-------------------")
                print("END staggered loop")
            
            self.H_inc.x.array[:] = self.H_new.x.array[:]
            #self.mpi_print(f"alpha.max :  {self.alpha.x.array.max()}")
            if self.alpha.x.array.max() > 1:
                break

            #Calculate force and energy + save data in txt ad xdmf format
            self.save_data(it, t)
            mgis_bv.update(self.m)
            t_old = t
            it +=1
            

    def save_data(self, step_nb, time):

        # save load
        self.loading.append([step_nb,time, self.load_func.x.array[0]])
        
        # save force
        ut.local_project(self.msh, self.sig_quad, self.V_dg, self.sig_dg)
        f = fem.assemble_scalar(fem.form(self.sig_dg[0]*self.ds( self.boundary_indices["right"])))
        fn = fem.assemble_scalar(fem.form(self.sig_dg[1]*self.ds( self.boundary_indices["right"])))
        ft = fem.assemble_scalar(fem.form(self.sig_dg[3]*self.ds( self.boundary_indices["right"])))
        self.reaction.append([step_nb,f,fn,ft])
        
        # save elastic energy and fracture energy
        E_el_form      = fem.form(0.5 * ufl.inner(self.sig_quad, bv.epsilon_mfront(self.u)) * self.dxm)
        E_el = fem.assemble_scalar(E_el_form)

        E_frac_form     = fem.form( Gc/cw*(bv.w(self.alpha)/lc + lc * ufl.dot(ufl.grad(self.alpha), ufl.grad(self.alpha))) * self.dx)
        E_frac= fem.assemble_scalar(E_frac_form)

        E_kin_form     = fem.form(0.5 * self.rho*ufl.inner(self.u_p, self.u_p) * self.dx)
        E_kin          = fem.assemble_scalar(E_kin_form)
        self.energie.append([step_nb, E_el, E_frac, E_kin])
        
        np.savetxt(self.savedir + 'loading.txt',np.array(self.loading) )                     
        np.savetxt(self.savedir + 'reaction.txt',np.array(self.reaction) )
        np.savetxt(self.savedir + 'energie.txt', np.array(self.energie) )

        # check strain from fenicsx project on DG-space
        eps_expr = fem.Expression(bv.epsilon_mfront(self.u), self.V_dg.element.interpolation_points())
        self.eps_dg.interpolate(eps_expr)
       # ut.local_project(self.msh, self.eps_quad, self.V_dg, self.eps_dg)

        if comm.rank == 0:
            #print('(Fx, Fy, F): (%g, %g, %g)'%(Fx,Fy, F))
            print("\nf, fn, ft: (%g, %g, %g)"%(f,fn, ft))
            print("\nE_el: %0.3g, E_kin : %0.3g, E_frac : %0.3g"%(E_el, E_kin, E_frac))
            print("\nalpha_CG.max:", (self.alpha.x.array.max()))
            #print('\nalpha_QUAD.max:', self.alpha_mfront.x.array.max())
            print('\nalpha_CG.min:', self.alpha.x.array.min())

        # export to paraview
        if step_nb % 1 == 0:
            self.xdmf0.write_function(self.u, time)
            self.xdmf1.write_function(self.alpha, time)
            self.xdmf4.write_function(self.u_p, time)
            self.xdmf5.write_function(self.u_pp, time)
            if self.save_stress:
                print("\n\n--save stress--")
                self.xdmf2.write_function(self.sig_dg, time)
            if self.save_strain:
                print("\n\n--save strain--")
                self.xdmf3.write_function(self.eps_dg, time)
        
        
        
                             
        
        

if __name__ == '__main__':
    

    dynamic_reg = True
    savedir        = "r_Verlet_3mps_t1_csec0_9_MPI_2proc/"
    print("Orthogonal spectral - Ngyuen")
    comm = MPI.COMM_WORLD
    Lx, Ly = .1, .01 # Dimensions of the beam
    nx, ny = 20, 2
    
    msh = mesh.create_rectangle(comm, [[0.,0.],[Lx, Ly]], [nx, ny], mesh.CellType.quadrilateral, ghost_mode=cpp.mesh.GhostMode.none)

    ndim = msh.topology.dim
    # notice that cpp.mesh.h(self.msh, self.ndim, [0])[0] gives the diagonal size of the square cell
    #h_cell = np.sqrt(2)*cpp.mesh.h(msh, ndim, [0])[0]
    h_min = cpp.mesh.h(msh, ndim, [0])[0]

    # setup material model
    E, nu = PETSc.ScalarType(190000000000), PETSc.ScalarType(0.2) #[Pa]
    mu    = fem.Constant(msh,E / (2.0*(1.0 + nu))) # [Pa]
    lmbda = fem.Constant(msh,E * nu / ((1.0 + nu)* (1.0 - 2.0*nu)))
    K0 = lmbda+2/3*mu
    lc = 2.5 * h_min
    Gc = PETSc.ScalarType(22130)#N/m
    cw = PETSc.ScalarType(2.0)



    # threshold to obtain a linear elastic phase
    m = 0.00
    phi_c = m*Gc/lc
    material_parameters = {"E":E, "nu":nu, "mu":mu, "lmbda":lmbda, "K0":K0, "lc":lc, "Gc":Gc, "cw":cw, "threshold": phi_c }

    if dynamic_reg :
        # dynamic parameters
        rho       = PETSc.ScalarType(8000.0)  # kg/m³
        celerity  = np.sqrt(E/rho)   #  3700 m/s
        dt_crit   = h_min/np.sqrt(E/rho) # => 2,703e-7s 0.27 µs
        eta = 0.01
        rho_d       = 2*Gc*lc/celerity**2
        material_parameters = {"E":E, "nu":nu, "mu":mu, "lmbda":lmbda, "K0":K0, "lc":lc, "Gc":Gc, "cw":cw, "threshold": phi_c, "rho":rho, "celerity":celerity, "viscosity" : eta, "rho_damage":rho_d}
        if comm.rank == 0 :
            print("\nwave propagation {:.2e}m/s".format(celerity))
            print("Pas de temps critique {:.2e}s".format(dt_crit))
            print("Micro damage inertia {:.2e}s".format(rho_d))
        
        
    
    #behavior = bv.ElasticQSModel(ndim, material_parameters)
    behavior = bv.AmorModel(ndim, material_parameters)
    
    fracture_pb = FractureProblem(msh,  behavior, material_parameters,savedir )
    nb_cell = msh.topology.index_map(2).size_local
    nb_node_topo = msh.topology.index_map(0).size_local
    nb_node_geom = len(msh.geometry.x)
    fracture_pb.mpi_print(f'nb_cell : {nb_cell}')
    fracture_pb.mpi_print(f'nnode (with ghost node) : {nb_node_geom}')
    
    



    # solve problem
    #fracture_pb.set_time_stepping()
    #fracture_pb. set_initilization()
    #fracture_pb.set_variational_formulation()

    fracture_pb.solve_explicit()
    
    #fracture_pb.solve_qs()    


