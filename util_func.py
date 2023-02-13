import ufl
from dolfinx import fem

from mpi4py import MPI
from petsc4py import PETSc


degS = 2
metadata_quad = {"quadrature_degree": degS, "quadrature_scheme": "default"}
dxm = ufl.dx(metadata=metadata_quad)

def local_project(msh, v, V, u=None):
    """
    projects v on V with custom quadrature scheme dedicated to
    FunctionSpaces V of `Quadrature` type

    if u is provided, result is appended to u
    """
    x = ufl.SpatialCoordinate(msh)
    dv, v_ = ufl.TrialFunction(V), ufl.TestFunction(V)
    a_proj = fem.form(ufl.inner(dv, v_)*dxm)
    L_proj = fem.form(ufl.inner(v, v_)*dxm)

    b = fem.petsc.assemble_vector(L_proj)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    A = fem.petsc.assemble_matrix(a_proj)
    A.assemble()
    # Create CG Krylov solver and turn convergence monitoring on
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    #solver.setType('cg')
    pc = solver.getPC()
    solver.setType('preonly')
    pc.setType('lu')

    solver.setOperators(A)

    #solve
    if u is None:
        u = fem.Function(V)
        solver.solve(b, u.vector)
        #u.x.scatter_forward()
        return u
    else:
        solver.solve(b, u.vector)
        u.x.scatter_forward()
        return


metadata = {"quadrature_degree":4}
dx = ufl.Measure("dx", metadata=metadata)
def local_project_to_fenics(msh, v, V, u=None):
    """
    projects v on V with custom quadrature scheme dedicated to
    FunctionSpaces V of `Quadrature` type

    if u is provided, result is appended to u
    """
    x = ufl.SpatialCoordinate(msh)
    dv, v_ = ufl.TrialFunction(V), ufl.TestFunction(V)
    a_proj = fem.form(ufl.inner(dv, v_)*dx)
    L_proj = fem.form(ufl.inner(v, v_)*dx)

    b = fem.petsc.assemble_vector(L_proj)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    A = fem.petsc.assemble_matrix(a_proj)
    A.assemble()
    # Create CG Krylov solver and turn convergence monitoring on
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    #solver.setType('cg')
    pc = solver.getPC()
    solver.setType('preonly')
    pc.setType('lu')

    solver.setOperators(A)

    #solve
    if u is None:
        u = fem.Function(V)
        solver.solve(b, u.vector)
        #u.x.scatter_forward()
        return u
    else:
        solver.solve(b, u.vector)
        u.x.scatter_forward()
        return
