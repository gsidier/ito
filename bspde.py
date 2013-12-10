import __init__
import numpy
import ito.findiff as fd

class OptionPayoff(object):
	def expire(self, t, S):
		"""
		Input:
			t: single time of expiry
			S: vector of underlyer prices at expiry
		
		Return:
			V: vector of option prices at expiry
		"""
		raise NotImplemented
	
	def boundary_hi(self, t, Shi):
		"""
		Input:
			t: single time of expiry
			S: single underlyer price
		
		Return:
			V: single option price
		"""
		raise NotImplemented
	
	def boundary_lo(self, t, Slo):
		"""
		Input:
			t: single time of expiry
			S: single underlyer price
		
		Return:
			V: single option price
		"""
		raise NotImplemented
	
	def early_ex(self, t, S, Vhold):
		"""
		Input:
			t: single time
			S: vector of underlyer prices at expiry
			Vhold: vector of option price if not exercised
		
		Return:
			V: vector of option prices
		"""
		raise NotImplemented

class BSPde(object):
	"""
	Solve the Black-Scholes Pde.
	
	Stochastic model of underlying asset:
	
		dS / S = (r - b) dt + sigma dW
	
	r, b, sigma time varying.
	
		r: risk-free rate
		b: borrow rate
		sigma: volatility
	
	The Black-Scholes equation is 
		
		dV/dt + sigma2/2 S2 d2V/dS2 + (r - b) S dV/dS - r V = 0
	
	To solve this accurately we first apply the change of variables
	
		V = S0 u
		S = S0 exp(x)
		t = T - 2 tau / sigma2
		k = 2 r / sigma2
		k' = 2 b / sigma2
		
	which yields the equation
		
		du / dtau = - k u + (k - k' - 1) du / dx + d2u / dx2
	
	For computation of the vega we will bump the vol: sigma' = sigma + dsigma.
	However we keep the same grid for accuracy. This yields the equation
		
		du / dtau = - k c u + (k - k' - 1) c du / dx + c d2u / dx2
		
			c = (sigma' / sigma)2
			k = 2 r / sigma'2
			k' = 2 b / sigma'2
	"""
	
	def __init__(self, payoff, S, t, r, b, sigma, method = 'implicit', iteration = None):
		self.payoff = payoff
		self.S = S
		self.t = t
		self.r = r
		self.b = b
		self.sigma = sigma
		self.method = method
		self.iteration = iteration
		
		self.S0 = S[len(S) / 2]
		t0 = t[0]
		self.x = self.S_to_x(t0, S)
		self.T = t[-1]
		self.tau = (self.T - t) * self.sigma * self.sigma / 2
	
	def S_to_x(self, t, S):
		return numpy.log(S / self.S0) 
	
	def x_to_S(self, t, x):
		return self.S0 * numpy.exp(x)
	
	def u_to_V(self, u):
		return self.S0 * u
	
	def V_to_u(self, V):
		return 1. / self.S0 * V
	
	def solve(self, sigma = None):
		if sigma is None:
			sigma = self.sigma
		S = self.x_to_S(self.T, self.x)
		V = self.payoff.expire(self.T, self.S)
		u = self.V_to_u(V)
		C = sigma / self.sigma
		K = 2 * self.r / (sigma * sigma)
		K_ = 2 * self.b / (sigma * sigma)
		step = {
			'crank-nicolson': fd.solve_crank_nicolson,
			'explicit': fd.solve_explicit,
			'implicit': fd.solve_implicit
		}[self.method]

		def proj(t, S, u):
			Vhold = self.u_to_V(u)
			V = self.payoff.early_ex(t, S, Vhold)
			u = self.V_to_u(V)
			return u
		
		for (t, tau, dtau, c, k, k_) in reversed(zip(self.t[:-1], self.tau[:-1], self.tau[:-1] - self.tau[1:], C, K, K_)):
			S = self.x_to_S(t, self.x)
			L = - k * c + (k - k_ - 1) * c * fd.d_dx(self.x) + c * fd.d2_dx2(self.x)
			if self.iteration:
				solver = {
					'jacobi': fd.solve_jacobi,
					'gauss-seidel': fd.solve_gauss_seidel,
				}[self.iteration]
				
				if self.method == 'crank-nicolson':
					rhs = (1 + L * dt / 2)(u)
					A = (1 + L * dt / 2)
				elif self.method == 'implicit':
					rhs = u
					A = 1 + L * dt
				else:
					raise NotImplemented, "iteration not defined for fd scheme %s" % self.method
				
				u = solver(A, rhs, u, 1e-8, 100, lambda u: proj(t, S, u))
				
			else:
				u = step(L, dtau, u)
				u = proj(t, S, u)
		V = self.u_to_V(u)
		return V

class VanillaPayoff(object):
	
	def __init__(self, K, CP):
		self.K = K
		self.CP = str(CP)[0].upper()
		if self.CP not in "CP":
			raise ValueError, "Unrecognized call/put flag: " + CP
		
	def expire(self, t, S):
		if self.CP == 'C':
			V = (S - K)
		else:
			V = (K - S)
		V[V <= 0] = 0
		return V
	
	def boundary_hi(self, t, Shi):
		raise NotImplemented
	
	def boundary_lo(self, t, Slo):
		raise NotImplemented

class AmericanPayoff(VanillaPayoff):
	
	def early_ex(self, t, S, Vhold):
		V = self.expire(t, S)
		V[V < Vhold] = Vhold
		return V

class EuropeanPayoff(VanillaPayoff):
	
	def early_ex(self, t, S, Vhold):
		return Vhold

if __name__ == '__main__':
	
	S0 = 100.
	K = 100.
	CP = 'C'
	Nx = 101
	Nt = 101
	sns = numpy.linspace(-6, +6, Nx)
	sigma = .3 * numpy.ones(Nt)
	T = .25
	t = numpy.linspace(0, T, Nt)
	x = sns * sigma[0] * numpy.sqrt(T)
	S = S0 * numpy.exp(x)
	r = numpy.zeros(Nt)
	b = numpy.zeros(Nt)
	payoff = EuropeanPayoff(K, CP)
	pde = BSPde(payoff, S, t, r, b, sigma)
	V = pde.solve()
	
	from bs import black_scholes_1973
	Vref = black_scholes_1973(T, S, sigma[0], r[0], b[0], K, CP)['price']

