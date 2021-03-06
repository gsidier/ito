import numpy
import scipy.linalg

def l2(x):
	return numpy.sqrt(numpy.mean(x * x))
	
class FinDiffOp(object):
	"""
	Represents the tri-band linear finite difference operator
	
	[ d0 u0                 ]
	[ l0 d1 u1              ]
	[    l1 d2 u2           ]
	[         . . .         ]
	[        ln-2 dn-1 un-1 ]
	[             ln-1 dn   ]

	using the internal represenation:	

	self.bands = [ *  u0 u1 ... un-2 un-1 ]
                 [ d0 d1 d2 ... dn-1 dn   ]
                 [ l0 l1 l2 ... ln-1 *    ]
	"""
	
	inv = property(lambda self: FinDiffInvOp(self))
	
	def __init__(self, bands):
		self.bands = numpy.array(bands)
	
	@classmethod
	def Id(cls, n):
		return FinDiffOp([numpy.zeros(n), numpy.ones(n), numpy.zeros(n)])
	
	def set_boundary_lo(self, d0, u0):
		self.bands[0, 1] = u0
		self.bands[1, 0] = d0
	
	def set_boundary_hi(self, ln, dn):
		self.bands[2, -2] = ln
		self.bands[1, -1] = dn
	
	def __call__(self, y):
		shape = (1,) * (numpy.ndim(y) - 1) + (numpy.shape(y)[-1],)
		z = [ band.reshape(shape) * y for band in self.bands ]
		res = z[1]
		s = [ slice(None) ] * (z[0].ndim - 1)
		res[tuple(s + [ slice(None, -1) ])] += z[0][tuple(s + [ slice(1, None) ])]
		res[tuple(s + [ slice(1, None) ])] += z[2][tuple(s + [ slice(None, -1) ])]
		return res
	
	def __call__bak__(self, y):
		z = self.bands * y
		res = z[1, :]
		res[:-1] += z[0, 1:]
		res[1:] += z[2, :-1]
		return res
	
	def __rmul__(self, k):
		return FinDiffOp(k * self.bands)
	
	def __mul__(self, k):
		return FinDiffOp(self.bands * k)
	
	def __div__(self, k):
		return self * (1. / k)
	
	def __truediv__(self, k):
		return self.__div__(k)
	
	def __add__(self, op):
		return FinDiffOp(self.bands + op.bands)
	
	def __radd__(self, k):
		return k * self.Id(len(self.bands[1, :])) + self
	
	def __sub__(self, op):
		return FinDiffOp(self.bands - op.bands)

	def __rsub__(self, k):
		return k * self.Id(len(self.bands[1, :])) - self

class FinDiffInvOp(object):
	
	def __init__(self, fdop):
		self.fdop = fdop
	
	def __call__(self, y):
		return scipy.linalg.solve_banded((1, 1), self.fdop.bands, y)

def noboundary(l, d, u):
	l = numpy.hstack([l, [0, 0]])
	d = numpy.hstack([[0], d, [0]])
	u = numpy.hstack([[0, 0], u])
	bands = numpy.array([u, d, l])
	return FinDiffOp(bands)

def d_dx(x):
	"""
	Return the FinDiffOp that computes d/dx.
	
	x does not have to be evenly spaced. Three-point differences are computed
	with O(epsilon(h + k) + h3 + k3) precision where h is x[i] - x[i - 1] and k is x[i + 1] - x[i]
	and epsilon = |h - k|.
	
	The one-sided differentials are computed on the edge of the domain.
	"""
	# h - k = epsilon
	# f(x + h) = f + h f' + h2/2 f'' + h3/6 f''' + O(h4)
	# f(x - k) = f - k f' + k2/2 f'' - k3/6 f''' + O(k4)
	# f(x + h) / h2 - f(x - k) / k2 = (1 / h2 - 1 / k2) f + (1 / h + 1 / k) f' + 1/6 (h - k) f''' + O(h2 + k2)
	# (h + k) / hk f' = f(x + h) / h2 - f(x - k) / k2 - (1 / h2 - 1 / k2) f - 1/6 epsilon f''' + O(h2 + k2)
	# f' = k / (h (h + k)) f(x + h) - h / (k (h + k)) f(x - k) + (h - k) / hk f + O(epsilon (h + k) + h3 + k3)
	dx = x[1:] - x[:-1]
	h = dx[1:]
	k = dx[:-1]
	d = (h - k) / (k * h)
	u = k / (h * (h + k))
	l = - h / (k * (h + k))
	res = noboundary(l, d, u)
	res.set_boundary_lo(d0 = -1. / (x[1] - x[0]), u0 = +1. / (x[1] - x[0]))
	res.set_boundary_hi(dn = +1. / (x[-1] - x[-2]), ln = -1. / (x[-1] - x[-2]))
	return res
	
def d2_dx2(x):
	# h - k = epsilon
	# f(x + h) = f + h f' + h2/2 f'' + h3/6 f''' + O(h4)
	# f(x - k) = f - k f' + k2/2 f'' - k3/6 f''' + O(k4)
	# 1/h f(x + h) + 1/k f(x - k) = (1/h + 1/k) f + (h + k) / 2 f'' + 1/6 (h2 - k2) f''' + O(h4 + k4) 
	# f'' = 2 / (h (h + k)) f(x + h) + 2 / (k (h + k)) f(x -k) - 2 / hk f + O(epsilon + h3 + k3)
	dx = x[1:] - x[:-1]
	h = dx[1:]
	k = dx[:-1]
	d = - 2 / (h * k)
	u = 2 / (h * (h + k))
	l = 2 / (k * (h + k))
	return noboundary(l, d, u)

def solve_explicit(L, c, dt, y, ** bounds):
	# dy / dt = L y + c
	# (y2 - y1) / dt = L y1 + c
	# y2 = y1 + (L y1 + c) dt
	dy = dt * (L(y) + c)
	y2 = y + dy
	set_explicit_boundaries(y2, ** bounds)
	return y2

def set_explicit_boundaries(x, dirichlet = (None, None), neumann = (None, None), robin = (None, None)):
	"""
	x: the vector to which the boundary condition is applied
	dirichlet: (lo, hi) value at boundary
	neumann: (lo, hi) value of x[1] - x[0] or x[-1] - x[-2]
	robin: pairs ((a_lo, c_lo), (a_hi, c_hi)) such that:
		a_lo (u[1] + u[0]) / 2 + (u[1] - u[0]) + c_lo = 0
		a_hi (u[-2] + u[-1]) / 2 + (u[-1] - u[-2]) + c_hi = 0
	"""
	
	dirichlet_lo, dirichlet_hi = dirichlet
	if dirichlet_lo is not None:
		x[0] = dirichlet_lo
	if dirichlet_hi is not None:
		x[-1] = dirichlet_hi
	
	neumann_lo, neumann_hi = neumann
	if neumann_lo is not None:
		x[0] = x[1] - neumann_lo
	if neumann_hi is not None:
		x[-1] = x[-2] + neumann_hi
	
	robin_lo, robin_hi = robin
	if robin_lo is not None:
		a, c = robin_lo
		x[0] = ((1 + a * .5) * x[1] + c) / (1 - a * .5)
	if robin_hi is not None:
		a, c = robin_hi
		x[-1] = ((1 - a * .5) * x[-2] - c) / (1 + a * .5)

def solve_implicit(L, c, dt, y, ** bounds):
	# dy / dt = L y + c
	# (y2 - y1) / dt = L y2 + c
	# y2 - L y2 dt = y1 + c dt
	# y2 = (I - L dt)-1 (y1 + c dt)
	op = 1 - dt * L
	z = y + c * dt
	set_implicit_boundaries(op, z, ** bounds)
	y2 = op.inv(z) 
	return y2

def set_implicit_boundaries(A, b, dirichlet = (None, None), neumann = (None, None), robin = (None, None)):
	# A x = b
	# x[0] = x0
	# x[n] = xn
	# A[0, j] = 1{j == 0}
	# b[0] = x0
	# A[n, j] = 1{j == n}
	# b[n] = xn
	dirichlet_lo, dirichlet_hi = dirichlet
	if dirichlet_lo:
		A.set_boundary_lo(d0 = 1, u0 = 0)
		b[0] = dirichlet_lo
	if dirichlet_hi:
		A.set_boundary_hi(dn = 1, ln = 0)
		b[-1] = dirichlet_hi
	
	# A x = b
	# x[1] - x[0] = dx0
	# x[n] - x[n-1] = dxn
	# A[0, 0] = -1, A[0, 1] = 1, A[0, j] = 0 (j > 1)
	# b[0] = dx0
	# A[n, n-1] = -1, A[n, n] = 1, A[n, j] = 0 (j < n-1)
	# b[n] = dxn
	neumann_lo, neumann_hi = neumann
	if neumann_lo:
		A.set_boundary_lo(d0 = -1, u0 = 1)
		b[0] = neumann_lo
	if neumann_hi:
		A.set_boundary_hi(ln = -1, dn = 1)
		b[-1] = neumann_hi
	
def solve_crank_nicolson(L, c, dt, y, ** bounds):
	# dy / dt = L y + c
	# (y2 - y1) / dt = L (y1 + y2) / 2 + c
	# y1 + L y1 dt / 2 + c dt = y2 - L y2 dt / 2
	# y2 = (I - L dt / 2)-1 ((I + L dt / 2) y1 + c dt)
	
	op2 = 1 - L * dt / 2
	op1 = 1 + L * dt / 2
	z1 = op1(y)
	z2 = z1 + c * dt
	set_implicit_boundaries(op2, z2, ** bounds)
	y2 = op2.inv(z2)
	return y2

def iterate(step, x0, tol, maxiter, proj = None, callback = None):
	"""
	Perform a iteration step until convergence, or we run out of iterations.
	"""
	if proj:
		step1 = lambda x: proj(step(x))
	else:
		step1 = step
	for _ in range(maxiter):
		if callback:
			callback(locals())
		x = step1(x0)
		if l2(x - x0) <= tol:
			return x
		x0 = x
	raise RuntimeError("Convergence failed.")

def jacobi_step(d, UL, y, x0):
	return (y - UL(x0)) / d

def solve_jacobi(A, y, x0, tol, maxiter, proj = None, ** boundary):
	"""
	Given A a linear operator and y vector, solve A x = y for x by Jacobi iteration.
	"""
	ul = A.bands.copy()
	ul[1, :] = 0 
	UL = FinDiffOp(ul)
	d = A.bands[1, :]
	def project(x):
		x1 = proj(x) if proj else x
		set_explicit_boundaries(x1, ** boundary)
		return x1
	return iterate(lambda x: jacobi_step(d, UL, y, x), x0, tol, maxiter, project)

def gauss_seidel_step(U, L, y, x0, ** boundary):
	rhs = y - L(x0)
	set_implicit_boundaries(U, rhs, ** boundary)
	return U.inv(rhs)
	
def solve_gauss_seidel(A, y, x0, tol, maxiter, proj = None, ** boundary):
	"""
	Given A a linear operator and y vector, solve A x = y for x by Gauss-Seidel iteration.
	"""
	l = A.bands.copy()
	l[0:2, :] = 0 
	L = FinDiffOp(l)
	u = A.bands.copy()
	u[2, :] = 0
	U = FinDiffOp(u)
	"""
	l = A.bands.copy()
	l[1:, :] = 0 
	L = FinDiffOp(l)
	u = A.bands.copy()
	u[0, :] = 0
	U = FinDiffOp(u)
	"""
	return iterate(lambda x: gauss_seidel_step(U, L, y, x, ** boundary), x0, tol, maxiter, proj)

if __name__ == '__main__':
	
	def l2(x):
		return numpy.sqrt(numpy.sum(x * x)) / len(x)
	
	def l1(x):
		return numpy.sum(numpy.abs(x)) / len(x)
	
	def linf(x):
		return numpy.max(numpy.abs(x))
	
	def eq(x, y, eps = 1e-10):
		return linf(x - y) < eps
	
	t = numpy.linspace(0, 1, 1001)
	x = t ** 2
	Dx = d_dx(x)
	D2x = d2_dx2(x)
	y = numpy.sin(x)
	dy_dx = Dx(y)
	d2y_dx2 = D2x(y)
	assert(numpy.max(numpy.abs(dy_dx[1:-1] - numpy.cos(x)[1:-1])) < 1e-5)
	assert(numpy.max(numpy.abs(d2y_dx2[1:-1] + numpy.sin(x)[1:-1])) < 1e-5)
	y = numpy.exp(x)
	dy_dx = Dx(y)
	d2y_dx2 = D2x(y)
	assert(numpy.max(numpy.abs(dy_dx[1:-1] - numpy.exp(x)[1:-1])) < 1e-5)
	assert(numpy.max(numpy.abs(d2y_dx2[1:-1] - numpy.exp(x)[1:-1])) < 1e-5)
	
	# Solve the heat equation
	# dy/dt = 1/2 d2y/dx2
	def phi(x, sigma):
		return numpy.exp(- x * x / (2 * sigma * sigma)) / sigma
	x = numpy.linspace(-9, 9, 361)
	sigma = 1
	y = phi(x, sigma)
	T = numpy.linspace(1, 2, 401)
	L = d2_dx2(x) / 2
	Yt = [ y ]
	# explicit
	for (t, dt) in zip(T[1:], T[1:] - T[:-1]):
		sigma_t = sigma * numpy.sqrt(t)
		#L.set_boundary_lo(d0 = .5 * (x[-1] ** 2 / sigma_t ** 4 - 1 / sigma_t ** 2), u0 = 0)
		#L.set_boundary_hi(dn = .5 * (x[0] ** 2 / sigma_t ** 4 - 1 / sigma_t ** 2), ln = 0)
		y = solve_explicit(L, 0, dt, y)
		Yt.append(y)
		u = phi(x, sigma_t)
		assert eq(y, u, 1e-3)
		assert l2(y - u) < 1e-4
	print("explicit:")
	print("l1:", l1(y - u))
	print("l2:", l2(y - u))
	print("linf:", linf(y - u))
	print()
	# implicit
	y = phi(x, sigma)
	Yt = [ y ]
	for (t, dt) in zip(T[1:], T[1:] - T[:-1]):
		sigma_t = sigma * numpy.sqrt(t)
		#L.set_boundary_lo(d0 = .5 * (x[-1] ** 2 / sigma_t ** 4 - 1 / sigma_t ** 2), u0 = 0)
		#L.set_boundary_hi(dn = .5 * (x[0] ** 2 / sigma_t ** 4 - 1 / sigma_t ** 2), ln = 0)
		y = solve_implicit(L, 0, dt, y)
		Yt.append(y)
		u = phi(x, sigma_t)
		assert eq(y, u, 1e-3)
		assert l2(y - u) < 1e-4
	print("implicit:")
	print("l1:", l1(y - u))
	print("l2:", l2(y - u))
	print("linf:", linf(y - u))
	print()
	# crank-nicolson
	y = phi(x, sigma)
	Yt = [ y ]
	for (t, dt) in zip(T[1:], T[1:] - T[:-1]):
		sigma_t = sigma * numpy.sqrt(t)
		#L.set_boundary_lo(d0 = .5 * (x[-1] ** 2 / sigma_t ** 4 - 1 / sigma_t ** 2), u0 = 0)
		#L.set_boundary_hi(dn = .5 * (x[0] ** 2 / sigma_t ** 4 - 1 / sigma_t ** 2), ln = 0)
		y = solve_crank_nicolson(L, 0, dt, y)
		Yt.append(y)
		u = phi(x, sigma_t)
		assert eq(y, u, 1e-3)
		assert l2(y - u) < 1e-4
	print("crank-nicolson:")
	print("l1:", l1(y - u))
	print("l2:", l2(y - u))
	print("linf:", linf(y - u))
	print()
	# solve implicit equation by jacobi
	y = phi(x, sigma)
	dt = .01
	y2 = solve_jacobi(1 - L * dt, y, y, 1e-8, 100) # y2 = (I - L dt)-1 y
	print("jacobi: success")
	# solve implicit equation by gauss-seidel
	y = phi(x, sigma)
	dt = .01
	y2 = solve_gauss_seidel(1 - L * dt, y, y, 1e-8, 100) # y2 = (I - L dt)-1 y
	print("gauss-seidel: success")
	# solve crank-nicolson equation by jacobi
	A = 1 - L * dt / 2
	rhs = (1 + L * dt / 2)(y)
	y = phi(x, sigma)
	dt = .01
	y2 = solve_jacobi(A, rhs, y, 1e-8, 100) # y2 = (I - L dt)-1 y
	print("jacobi: success")
	print("l2 err", l2(A(y2) - rhs))
	# solve crank-nicolson equation by gauss-seidel
	y = phi(x, sigma)
	dt = .01
	y2 = solve_gauss_seidel(A, rhs, y, 1e-8, 100) # y2 = (I - L dt)-1 y
	print("gauss-seidel: success")
	print("l2 err", l2(A(y2) - rhs))
	"""
	# iterative solver for implicit scheme
	y = phi(x, sigma)
	dt = .01
	y2 = solve_implicit(L, 0, dt, y) # y2 = (I - L dt)-1 y
	y0 = y
	yn = y0
	#import pdb; pdb.set_trace()
	while 1:
		print('.',)
		yn = y0 + (L * dt)(yn)
		yn[0] = yn[1] - (x[1] - x[0]) * (yn[2] - yn[1]) / (x[2] - x[1])
		yn[-1] = yn[-2] + (x[-1] - x[-2]) * (yn[-2] - yn[-3]) / (x[-2] - x[-3]) 
		if l2(yn - y2) < 1e-7:
			break
	print
	# iterative solver for crank-nicolson scheme
	L.set_boundary_lo(d0 = .5 * (x[-1] ** 2 / sigma_t ** 4 - 1 / sigma_t ** 2), u0 = 0)
	L.set_boundary_hi(dn = .5 * (x[0] ** 2 / sigma_t ** 4 - 1 / sigma_t ** 2), ln = 0)
	y = phi(x, sigma)
	dt = .1
	y2 = solve_crank_nicolson(L, 0, dt, y) # y2 = (I - L dt / 2)-1 (I + L dt / 2) y
	y0 = (1 + L * dt / 2)(y)
	yn = y0
	while 1:
		import pdb; pdb.set_trace()
		print('C',)
		yn = y0 + (L * dt / 2)(yn)
		if l2(yn - y2) < 1e-7:
			break
	print
	"""
