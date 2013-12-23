from ito.bspde import BSPde, AmericanPayoff
import numpy

if __name__ == '__main__':
	
	def l1(x):
		return numpy.mean(numpy.abs(x))
	
	def l2(x):
		return numpy.sqrt(numpy.mean(x * x))
	
	def linf(x):
		return numpy.max(numpy.abs(x))
	
	S0 = 10.
	K = 10.
	Nx = 101
	Nt = 101
	sns = numpy.linspace(-6, +6, Nx)
	sigma = .3
	T = .25
	t = numpy.linspace(0, T, Nt)
	x = sns * sigma * numpy.sqrt(T)
	r = 0.02
	b = 0
	divs = [(1. / 12., .10)]

	D = sum(numpy.exp(- r * ti) * di for (ti, di) in divs)
	X0 = S0 - D
	X = X0 * numpy.exp(x)
	S = X + D
	
	nexpl = 20
	errmax = .6e-2
	
	iteration = 'jacobi'
	scheme = 'implicit'
	
	full_grid = 1
	
	call = AmericanPayoff(K, 'C')
	pde = BSPde(call, S, t, r, b, divs, sigma, scheme, iteration = iteration)
	
	res = pde.solve(outputs = [ 'full_grid' ])
	Vt = res['price']
	V = Vt[0, :]
		
