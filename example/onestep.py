from ito.bspde import BSPde, EuropeanPayoff
from ito.bs import black_scholes_1973
import numpy

def days(n):
	return n / 365.

def months(n):
	return n / 12.

def pct(x):
	return 1e-2 * x

def bp(x):
	return 1e-4 * x

if __name__ == '__main__':
	
	def l1(x):
		return numpy.mean(numpy.abs(x))
	
	def l2(x):
		return numpy.sqrt(numpy.mean(x * x))
	
	def linf(x):
		return numpy.max(numpy.abs(x))
	
	CP = 'C'
	S0 = 10.
	K = 10.
	Nx = 101
	Nt = 2
	sns = numpy.linspace(-6, +6, Nx)
	sigma = pct(30)
	T = days(1)
	t = numpy.linspace(0, T, Nt)
	x = sns * sigma * numpy.sqrt(T)
	r = pct(2)
	b = bp(0)
	divs = []

	D = sum(numpy.exp(- r * ti) * di for (ti, di) in divs)
	X0 = S0 - D
	X = X0 * numpy.exp(x)
	S = X + D
	
	nexpl = 20
	errmax = .6e-2
	
	iteration = None
	scheme = 'implicit'
	
	opt = EuropeanPayoff(K, CP)
	pde = BSPde(opt, S, t, r, b, divs, sigma, scheme, iteration = iteration)
	
	res = pde.solve(outputs = [ 'full_grid' ])
	Vt = res['price']
	V = Vt[0, :]
	
	intrinsic = opt.expire(T, S)
	premium = V - intrinsic
	
	ref = black_scholes_1973(T, S, sigma, r, b, K, CP)
	ref_premium = ref - intrinsic
