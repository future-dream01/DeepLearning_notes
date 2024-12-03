import sympy

v=sympy.symbols("v")
eq=sympy.Eq(4*v**2,29*v+8)
out=sympy.nsolve(eq,v,1)
print(out)