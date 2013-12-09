import numpy
import scipy.linalg

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
	
	def set_boundary_hi(self, d0, u0):
		self.bands[0, 1] = u0
		self.bands[1, 0] = d0
	
	def set_boundary_lo(self, ln, dn):
		self.bands[2, -2] = ln
		self.bands[1, -1] = dn
	
	def __call__(self, y):
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
	return noboundary(l, d, u)
	
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

def solve_explicit(L, dt, y):
	# dy / dt = L y
	# (y2 - y1) / dt = L y1
	# y2 = y1 + L y1 dt
	dy = dt * L(y)
	y2 = y + dy
	return y2

def solve_implicit(L, dt, y):
	# dy / dt = L y
	# (y2 - y1) / dt = L y2
	# y2 - L y2 dt = y1
	# y2 = (I - L dt)-1 y1
	op = 1 - dt * L
	y2 = op.inv(y)
	return y2

def solve_crank_nicolson(L, dt, y):
	# dy / dt = L y
	# (y2 - y1) / dt = L (y1 + y2) / 2
	# y1 + L y1 dt / 2 = y2 - L y2 dt / 2
	# y2 = (I - L dt / 2)-1 (I + L dt / 2) y1
	op2 = 1 - L * dt / 2
	op1 = 1 + L * dt / 2
	z = op1(y)
	y2 = op2.inv(z)
	return y2

def solve_jacobi(A, y, x0, tol, maxiter):
	"""
	Given A a linear operator and y vector, solve A x = y for x by Jacobi iteration.
	"""
	ul = A.bands.copy()
	ul[1, :] = 0 
	UL = FinDiffOp(ul)
	d = A.bands[1, :]
	x = x0
	for _ in xrange(maxiter):
		print ".", 
		x = (y - UL(x)) / d
		if l2(x - x0) <= tol:
			return x
		x0 = x
	raise RuntimeError, "jacobi failed to converge after %d iterations" % maxiter

def jacobi_step(d, UL, y, x0):
	return (y - UL(x0)) / d

def gauss_seidel_step(U, L, y, x0):
	return U.inv(y - L(x0))

def iterate(step, x0, tol, maxiter):
	"""
	Perform a iteration step until 
	"""
	x = x0
	for _ in xrange(maxiter):
		x = step(x)
		if l2(x - x0) <= tol:
			return x
	raise RuntimeError, "Convergence failed."
	
def gauss_seidel(A, y, x0, tol, maxiter):
	"""
	Given A a linear operator and y vector, solve A x = y for x by Gauss-Seidel iteration.
	"""
	l = A.bands.copy()
	l[0:2, :] = 0 
	L = FinDiffOp(l)
	u = A.bands.copy()
	u[2, :] = 0
	U = FinDiffOp(u)
	x = x0
	for _ in xrange(maxiter):
		print ".", 
		x = U.inv(y - L(x))
		if l2(x - x0) <= tol:
			return x
		x0 = x
	raise RuntimeError, "gauss-seidel failed to converge after %d iterations" % maxiter

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
		#L.set_boundary_hi(d0 = .5 * (x[-1] ** 2 / sigma_t ** 4 - 1 / sigma_t ** 2), u0 = 0)
		#L.set_boundary_lo(dn = .5 * (x[0] ** 2 / sigma_t ** 4 - 1 / sigma_t ** 2), ln = 0)
		y = solve_explicit(L, dt, y)
		Yt.append(y)
		u = phi(x, sigma_t)
		assert eq(y, u, 1e-3)
		assert l2(y - u) < 1e-4
	print "explicit:"
	print "l1:", l1(y - u)
	print "l2:", l2(y - u)
	print "linf:", linf(y - u)
	print
	# implicit
	y = phi(x, sigma)
	Yt = [ y ]
	for (t, dt) in zip(T[1:], T[1:] - T[:-1]):
		sigma_t = sigma * numpy.sqrt(t)
		#L.set_boundary_hi(d0 = .5 * (x[-1] ** 2 / sigma_t ** 4 - 1 / sigma_t ** 2), u0 = 0)
		#L.set_boundary_lo(dn = .5 * (x[0] ** 2 / sigma_t ** 4 - 1 / sigma_t ** 2), ln = 0)
		y = solve_implicit(L, dt, y)
		Yt.append(y)
		u = phi(x, sigma_t)
		assert eq(y, u, 1e-3)
		assert l2(y - u) < 1e-4
	print "implicit:"
	print "l1:", l1(y - u)
	print "l2:", l2(y - u)
	print "linf:", linf(y - u)
	print
	# crank-nicolson
	y = phi(x, sigma)
	Yt = [ y ]
	for (t, dt) in zip(T[1:], T[1:] - T[:-1]):
		sigma_t = sigma * numpy.sqrt(t)
		#L.set_boundary_hi(d0 = .5 * (x[-1] ** 2 / sigma_t ** 4 - 1 / sigma_t ** 2), u0 = 0)
		#L.set_boundary_lo(dn = .5 * (x[0] ** 2 / sigma_t ** 4 - 1 / sigma_t ** 2), ln = 0)
		y = solve_crank_nicolson(L, dt, y)
		Yt.append(y)
		u = phi(x, sigma_t)
		assert eq(y, u, 1e-3)
		assert l2(y - u) < 1e-4
	print "crank-nicolson:"
	print "l1:", l1(y - u)
	print "l2:", l2(y - u)
	print "linf:", linf(y - u)
	print
	# solve implicit equation by jacobi
	y = phi(x, sigma)
	dt = .01
	y2 = solve_jacobi(1 - L * dt, y, y, 1e-8, 100) # y2 = (I - L dt)-1 y
	print "jacobi: success"
	# solve implicit equation by gauss-seidel
	y = phi(x, sigma)
	dt = .01
	y2 = gauss_seidel(1 - L * dt, y, y, 1e-8, 100) # y2 = (I - L dt)-1 y
	print "gauss-seidel: success"
	# solve crank-nicolson equation by jacobi
	y = phi(x, sigma)
	dt = .01
	y2 = solve_jacobi(1 - L * dt / 2, (1 + L * dt / 2)(y), y, 1e-8, 100) # y2 = (I - L dt)-1 y
	print "jacobi: success"
	# solve crank-nicolson equation by gauss-seidel
	y = phi(x, sigma)
	dt = .01
	y2 = gauss_seidel(1 - L * dt / 2, (1 + L * dt / 2)(y), y, 1e-8, 100) # y2 = (I - L dt)-1 y
	print "gauss-seidel: success"
	"""
	# iterative solver for implicit scheme
	y = phi(x, sigma)
	dt = .01
	y2 = solve_implicit(L, dt, y) # y2 = (I - L dt)-1 y
	y0 = y
	yn = y0
	#import pdb; pdb.set_trace()
	while 1:
		print '.',
		yn = y0 + (L * dt)(yn)
		yn[0] = yn[1] - (x[1] - x[0]) * (yn[2] - yn[1]) / (x[2] - x[1])
		yn[-1] = yn[-2] + (x[-1] - x[-2]) * (yn[-2] - yn[-3]) / (x[-2] - x[-3]) 
		if l2(yn - y2) < 1e-7:
			break
	print
	# iterative solver for crank-nicolson scheme
	L.set_boundary_hi(d0 = .5 * (x[-1] ** 2 / sigma_t ** 4 - 1 / sigma_t ** 2), u0 = 0)
	L.set_boundary_lo(dn = .5 * (x[0] ** 2 / sigma_t ** 4 - 1 / sigma_t ** 2), ln = 0)
	y = phi(x, sigma)
	dt = .1
	y2 = solve_crank_nicolson(L, dt, y) # y2 = (I - L dt / 2)-1 (I + L dt / 2) y
	y0 = (1 + L * dt / 2)(y)
	yn = y0
	while 1:
		import pdb; pdb.set_trace()
		print 'C',
		yn = y0 + (L * dt / 2)(yn)
		if l2(yn - y2) < 1e-7:
			break
	print
	"""
