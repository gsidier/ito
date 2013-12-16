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
	
	def __init__(self, payoff, S, t, r, b, divs, sigma, method = 'implicit', iteration = None):
		"""
		S: length Nx array of grid stock prices
		t: length Nt array of grid times
		r: scalar or length Nt - 1 array of rates: r[i] = rate in interval (t[i], t[i+1])
		b: scalar or length Nt - 1 array of borrow rate: b[i] = borrow in (t[i], t[i+1])
		divs: array of (ti, di) dividends ti: dividend date, di: dividend amt
		sigma: scalar or array of volatilities: sigma[i] = volatility in (t[i], t[i+1])
		method: one of 'implicit', 'explicit', 'crank-nicolson'
		iteration: None or one of 'jacobi', 'gauss-seidel', 'sor'
		"""
		self.Nx = len(S)
		self.Nt = len(t)
		self.payoff = payoff
		self.S = S
		self.t = t
		self.r = r + numpy.zeros(self.Nt - 1)
		self.b = b + numpy.zeros(self.Nt - 1)
		self.divs = divs
		self.sigma = sigma + numpy.zeros(self.Nt - 1)
		if isinstance(method, (str, unicode)):
			self.method = [ method ] * (Nt - 1)
		else:
			self.method = method
		self.iteration = iteration
		self.discount = numpy.insert(numpy.cumprod([numpy.exp(- rt * dt) for (rt, dt) in zip(self.r[:len(self.t) - 1], self.t[1:] - self.t[:-1]) ]
				), 0, 1.) ## numpy.exp(- r * t)
		
		# X = S + D
		# Dt = sum[ti >= t] B(t, ti) * di
		divs_t = [ ] # divs_t[j] = sum(B(t, ti) * di for (di, ti) in divs if t[j] <= ti < t[j+1])
		divs = [(ti, di) for (ti, di) in divs if ti >= self.t[0]]
		divs = divs[::-1]
		for (disc, t1, t2) in zip(self.discount, self.t, list(self.t[1:]) + [ None ] ):
			x = 0
			while divs:
				ti, di = divs[-1]
				if (ti >= t1) and (t2 is None) or (ti < t2):
					x += disc * di
					divs.pop()
				else:
					break
			divs_t.append(x)
		sumdivs_t = numpy.cumsum(divs_t[::-1])[::-1]
		self.D = sumdivs_t / self.discount
		
		self.S0 = S[len(S) / 2]
		t0 = t[0]
		self.x = self.S_to_x(t0, S)
		self.T = t[-1]
		dtau = (t[1:] - t[:-1]) * self.sigma * self.sigma / 2
		self.tau = numpy.append(numpy.cumsum(dtau[::-1])[::-1], 0)
	
	def S_to_x(self, t, S):
		return numpy.log(S / self.S0) 
	
	def x_to_S(self, t, x):
		return self.S0 * numpy.exp(x)
	
	def u_to_V(self, u):
		return self.S0 * u
	
	def V_to_u(self, V):
		return 1. / self.S0 * V
	
	def solve(self, sigma = None, outputs = [ ]):
		"""
		sigma: None or scalar or vector of vols length Nt - 1
		outputs: list of optional outputs:
			-	price
			-	delta
			-	gamma
			-	vega
			-	theta
			-	rho
			-	rho_borrow
			-	vanna
			-	volga
			-	full_grid
			-	exercise_boundary
		"""
		res = { }
		if sigma is None:
			sigma = self.sigma
		else:
			sigma = sigma + numpy.zeros(self.Nt - 1)
		S = self.x_to_S(self.T, self.x)
		V = self.payoff.expire(self.T, self.S)
		u = self.V_to_u(V)
		C = sigma / self.sigma
		K = 2 * self.r / (sigma * sigma)
		K_ = 2 * self.b / (sigma * sigma)
		stepsmap = {
			'crank-nicolson': fd.solve_crank_nicolson,
			'explicit': fd.solve_explicit,
			'implicit': fd.solve_implicit
		}
		steps = [ stepsmap[meth] for meth in self.method ]

		def proj(t, S, u):
			Vhold = self.u_to_V(u)
			V = self.payoff.early_ex(t, S, Vhold)
			u = self.V_to_u(V)
			return u
		
		Vt = [ ]
		const = numpy.zeros(Nx)
		
		for (t, tau, dtau, c, k, k_, disc_t, step) in reversed(zip(self.t[:-1], self.tau[:-1], self.tau[:-1] - self.tau[1:], C, K, K_, self.discount, steps)):
			S = self.x_to_S(t, self.x)
			L = - k * c + (k - k_ - 1) * c * fd.d_dx(self.x) + c * fd.d2_dx2(self.x)
			const[:] = 0
			B_t_T = self.discount[-1] / disc_t
			if self.payoff.CP == 'C':
				# du/dtau = dV/dtau / S0 = 1/S0 d(S - K B(t, T))dt dt/dtau = - K dB(t, T)/dt dt/dtau
				# => du = - 1/S0 K dB(t, T)/dtau dtau
				# dB/dtau = - dr/dtau B = - k B
				# => du = k B K
				L.set_boundary_lo(0, 0)
				L.set_boundary_hi(0, 0)
				const[-1] = k * B_t_T * self.payoff.K / S0
			else:
				# u = K B(t, T) - S
				# du = - k K
				L.set_boundary_lo(0, 0)
				L.set_boundary_hi(0, 0)
				const[0] = - k * self.payoff.K / S0
			
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
				u = step(L, const, dtau, u)
				u = proj(t, S, u)
			
			V = self.u_to_V(u)
			if 'full_grid' in outputs:
				Vt.append(V)
		
		V = self.u_to_V(u)
		res['price'] = V
		if 'full_grid' in outputs:
			Vt = Vt[::-1]
			res['full_grid'] = Vt
		
		if outputs:
			return res
		else:
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
	
	def l1(x):
		return numpy.mean(numpy.abs(x))
	
	def l2(x):
		return numpy.sqrt(numpy.mean(x * x))
	
	def linf(x):
		return numpy.max(numpy.abs(x))
	
	S0 = 100.
	K = 100.
	CP = 'C'
	Nx = 101
	Nt = 101
	sns = numpy.linspace(-6, +6, Nx)
	sigma = .3
	T = .25
	t = numpy.linspace(0, T, Nt)
	x = sns * sigma * numpy.sqrt(T)
	S = S0 * numpy.exp(x)
	r = 0.02
	b = 0
	payoff = EuropeanPayoff(K, CP)
	nexpl = 20
	method = [ 'explicit' ] * nexpl + [ 'crank-nicolson' ] * (Nt - 1 - nexpl)
	#method = 'explicit'
	pde = BSPde(payoff, S, t, r, b, [], sigma, method)
	res = pde.solve(outputs = [ "full_grid" ])
	V = res['price']
	Vt = numpy.array(res['full_grid'])
	
	from bs import black_scholes_1973
	Vref = black_scholes_1973(T, S, sigma, r, b, K, CP)
	
	assert(l2(V - Vref) < .6e-2)

