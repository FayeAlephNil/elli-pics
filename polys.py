from sympy import symbols, Poly, degree, total_degree,sqrt,simplify, expand, Eq
import sympy as sp
import numpy.polynomial as npp
import numpy as np
import itertools
import random
from sympy.polys.multivariate_resultants import MacaulayResultant

def cubic_Delta(F,x,y,z):
    Fx = sp.expand(sp.diff(F, x))
    Fy = sp.expand(sp.diff(F, y))
    Fz = sp.expand(sp.diff(F, z))

    R = MacaulayResultant(polynomials=[Fx,Fy,Fz], variables=[x,y,z])
    return R,R.get_matrix().det()

def cubic_IJDelta(F, x, y, z):
    def attempt_quadratic_in_var(Fexpr, v, other1, other2):
        # treat F as polynomial in v
        Pv = Poly(Fexpr, v)
        degv = Pv.degree()
        if degv != 2:
            raise ValueError(f"Not quadratic in {v} (degree {degv}).")
        A = simplify(Pv.coeff_monomial(v**2))
        B = simplify(Pv.coeff_monomial(v**1))
        C = simplify(Pv.coeff_monomial(1))

        # R = B^2 - 4 A C (homogeneous polynomial in other1,other2 and s,t)
        R = expand(B**2 - 4*A*C)

        # Work in other1 (the variable we will shift) - ensure degree 3
        Rpoly = Poly(R, other1)
        degR = Rpoly.degree()
        if degR != 3:
            raise ValueError(f"R (as polynomial in {other1}) has degree {degR} (expected 3).")

        # coefficients of R in other1: c3*other1^3 + c2*other1^2*other2 + c1*other1*other2^2 + c0*other2^3
        c3 = simplify(Rpoly.coeff_monomial(other1**3))
        c2 = simplify(Rpoly.coeff_monomial(other1**2))
        # homogeneous shift: other1 = X1 + shift*other2  (shift is rational in s,t and maybe in x,z)
        shift = simplify(c2 / (3*c3))
        X1 = symbols('X1')
        Rshift = expand(R.subs(other1, X1 + shift*other2))
        Rshift_poly = Poly(Rshift, X1)

        # Now Rshift should be c3 * ( X1^3 + (d1/c3)*X1*other2^2 + (d0/c3)*other2^3 )
        # Let coeff_X1_in_Rshift = coefficient of X1^1 in Rshift (an expr in other2, s, t)
        coeff_X1 = simplify(Rshift_poly.coeff_monomial(X1**1))
        const_term = simplify(Rshift_poly.coeff_monomial(X1**0))

        # Extract normalized p_norm and q_norm:
        # After dividing by c3, coeff_X1/c3 = p_norm*other2^2, so p_norm = (coeff_X1/c3)/other2^2
        # and q_norm = (const_term/c3)/other2^3
        # But we will compute p_norm and q_norm first (these are the *normalized* invariants,
        # i.e. what you get if you divide the cubic by its leading coefficient).
        p_norm = simplify((coeff_X1 / c3) / (other2**2))
        q_norm = simplify((const_term / c3) / (other2**3))

        # Determine the original scalar lambda used to (possibly) normalize the cubic:
        # choose lambda_expr as the coefficient of other1**3 in the ORIGINAL cubic Fexpr
        # when viewed in (other1, other2). This is the scalar which if you had divided
        # the cubic by it would produce the normalized model we implicitly read p_norm/q_norm from.
        # We compute by writing Fexpr as polynomial in other1 (with other2 present) and taking that coeff.
        Fpoly_in_other1 = Poly(Fexpr, other1)
        lambda_expr = simplify(Fpoly_in_other1.coeff_monomial(other1**3))
        if lambda_expr == 0:
            # fallback: try coefficient of other2**3 (the complementary monomial)
            lambda_expr = simplify(Fpoly_in_other1.coeff_monomial(other2**3))
        if lambda_expr == 0:
            # as a last resort, try coefficient of other1^2*other2 etc (this is extremely rare)
            # We prefer a nonzero scalar so we can recover homogeneous invariants.
            raise ValueError("Could not find nonzero leading scalar lambda in the original cubic to rescale invariants.")

        # The true homogeneous invariants are obtained by multiplying back:
        # I_true = lambda_expr**4 * p_norm
        # J_true = lambda_expr**6 * q_norm
        I_st = simplify(lambda_expr**4 * p_norm)
        J_st = simplify(lambda_expr**6 * q_norm)
        Delta_st = simplify(4*I_st**3 - J_st**2)

        return (I_st, J_st, Delta_st)

    # Try variables in order y, x, z
    tries = [(y, x, z), (x, y, z), (z, x, y)]
    last_errs = []
    for v, o1, o2 in tries:
        try:
            return attempt_quadratic_in_var(F, v, o1, o2)
        except ValueError as e:
            last_errs.append((v, str(e)))
            continue

    msg = "Could not find a coordinate in which the cubic is quadratic. Tried y, x, z. Errors:\n"
    for v, err in last_errs:
        msg += f" - {v}: {err}\n"
    raise ValueError(msg)


def pencil_IJDelta(P,Q,x,y,z):
    s,t = symbols('s t')
    I_st, J_st, Delta_st = cubic_IJDelta(s*P+t*Q, x, y, z)
    return I_st, J_st, Delta_st

def example_nodal():
    x,y,z = symbols('x y z')
    example_nodall = Poly(z*y**2 - x**3, x,y,z)
    return example_nodall

def example_hex():
    x,y,z = symbols('x y z')
    example_hexx = Poly(z*y**2 - x**3 - z**3, x,y,z)
    return example_hexx

def example_square():
    x,y,z = symbols('x y z')
    return Poly(y**2*z - x**3 + x*z**2, x, y,z)

def rand_cubic(x,y,z):
    combos = [(i,j,k) for i,j,k in itertools.product(*([range(0,4)]*3)) if i+j+k == 3]
    F = 0
    for i,j,k in combos:
        if j != 3:
            coeff_real = random.uniform(-1,1)
            coeff_imag = random.uniform(-1,1)
            F += complex(coeff_real, coeff_imag) * x**i * y**j * z**k
    return F

# Gives zeros of polynomial on P1 as list of complex numbers + a flag indicating if [1 : 0] is a root
def find_zeros_on_P1(p):
    s,t = p.gens
    rev_lst_coeffs = p.subs({t: 1}).all_coeffs()[::-1]
    np_poly = npp.Polynomial([complex(a) for a in rev_lst_coeffs])
    
    return (np_poly.roots(), p.subs({s: 1, t: 0}) == 0)

def stereographic_proj(z,r=1):
    x = np.real(z)
    y = np.imag(z)
    t = ((r**2)*2)/(x**2+y**2+r**2)
    return [t*x,t*y,r*(1-t)]

