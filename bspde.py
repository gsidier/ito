import numpy
import ito.findiff as fd

def OptionPayoff(object):
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

def BSPde(object):
	"""
	Solve the Black-Scholes Pde.
	
	Stochastic model of underlying asset:
	
		dS / S = (r - b) dt + sigma dW
	
	r, b, sigma time varying.
	
		r: risk-free rate
		b: borrow rate
		sigma: volatility
	
	The Black-Scholes equation is 
		
		dV/dt + sigma2/2 S2 d2V/dt2 + (r - b) S dV/dt - r V = 0
	
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
	
	def __init__(self, payoff, S, t, r, b, sigma):
		self.payoff = payoff
		self.S = S
		self.t = t
		self.r = r
		self.b = b
		self.sigma = sigma
		
		self.S0 = S[len(S) / 2]
		self.x = self.S_to_x(S)
		self.T = t[-1]
		self.tau = (self.T - t) * self.sigma * self.sigma / 2
	
	def S_to_x(self, S):
		return numpy.log(S / self.S0) 
	
	def x_to_S(self, x):
		return self.S0 * numpy.exp(x)
	
	def u_to_V(self, u):
		return self.S0 * u
	
	def V_to_u(self, V):
		return 1. / self.S0 * V
	
	def solve(self, sigma):
		V = self.payoff.expire(self.T, self.S)
		u = self.V_to_u(V)
		C = sigma / self.sigma
		K = 2 * self.r / (sigma * sigma)
		K_ = 2 * self.b / (sigma * sigma)
		for (t, tau, dtau, c, k, k_) in reversed(zip(self.t[:-1], self.tau[:-1], self.tau[1:] - self.tau[:-1], C, K, K_)):
			L = - k * c + (k - k_ - 1) * c * fd.d_dx(self.x) + c * fd.d2_dx2(self.x)
			u = fd.solve_crank_nicolson(L, dtau, u)
			Vhold = self.u_to_V(u)

