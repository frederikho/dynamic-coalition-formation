import time
t0=time.time()
from juliacall import Main as jl
print(f"[{time.time()-t0:.0f}s] juliacall up", flush=True)
jl.seval('import Pkg; Pkg.add("AlgebraicSolving")')
print(f"[{time.time()-t0:.0f}s] AlgebraicSolving added", flush=True)
jl.seval('using AlgebraicSolving')
print(f"[{time.time()-t0:.0f}s] using AlgebraicSolving", flush=True)
# probe API on toy systems
jl.seval('R, (x,y) = polynomial_ring(QQ, ["x","y"])')
for desc, idl in [("zerodim x^2-1,y-x", "[x^2-1, y-x]"),
                  ("posdim y-x", "[y-x]"),
                  ("empty x, x-1", "[x, x-1]")]:
    jl.seval(f'I = Ideal({idl})')
    try:
        dim = jl.seval('dimension(I)')
    except Exception as e:
        dim = f"ERR {e}"
    try:
        rs = jl.seval('real_solutions(I)')
        rsr = repr(rs)
    except Exception as e:
        rsr = f"ERR {type(e).__name__}: {e}"
    print(f"{desc}: dimension={dim}  real_solutions={rsr}", flush=True)
print(f"[{time.time()-t0:.0f}s] done", flush=True)
