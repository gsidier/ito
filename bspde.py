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
		raise NotImplementedError
	
	def boundary(self, t, S, pde):
		"""
		Input:
			t: single time of expiry
			S: single underlyer price
			pde: a bspde object
		
		Return:
			{
				'dirichlet': (lo, hi),
				'neumann': (lo, hi)
			}
		"""
		raise NotImplementedError
	
	def early_ex(self, t, S, Vhold):
		"""
		Input:
			t: single time
			S: vector of underlyer prices at expiry
			Vhold: vector of option price if not exercised
		
		Return:
			V: vector of option prices
		"""
		raise NotImplementedError

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
		
		# S = X + D
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
		
		self.X0 = S[len(S) / 2] - self.D[0]
		t0 = t[0]
		self.x = self.S_to_x(0, S)
		self.T = t[-1]
		dtau = (t[1:] - t[:-1]) * self.sigma * self.sigma / 2
		self.tau = numpy.append(numpy.cumsum(dtau[::-1])[::-1], 0)
	
	def S_to_x(self, i, S):
		return self.X_to_x(S - self.D[i]) 
	
	def x_to_S(self, i, x):
		return self.x_to_X(x) + self.D[i]
	
	def X_to_x(self, X):
		return numpy.log(X / self.X0)
	
	def x_to_X(self, x):
		return self.X0 * numpy.exp(x)
	
	def u_to_V(self, u):
		return self.X0 * u
	
	def V_to_u(self, V):
		return 1. / self.X0 * V
	
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
		S = self.x_to_S(-1, self.x)
		V = self.payoff.expire(self.T, S)
		u = self.V_to_u(V)
		C = sigma / self.sigma
		K = 2 * self.r / (sigma * sigma)
		K_ = 2 * self.b / (sigma * sigma)
		stepsmap = {
			'crank-nicolson': fd.solve_crank_nicolson,
			'explicit': fd.solve_explicit,
			'implicit': fd.solve_implicit
		}

		def proj(t, S, u):
			Vhold = self.u_to_V(u)
			V = self.payoff.early_ex(t, S, Vhold)
			u = self.V_to_u(V)
			return u
		
		Vt = [ ]
		
		for (i, t, tau, dtau, c, k, k_, disc_t, method) in reversed(zip(range(self.Nt - 1), self.t[:-1], self.tau[:-1], self.tau[:-1] - self.tau[1:], C, K, K_, self.discount, self.method)):
			S = self.x_to_S(i, self.x)
			L = - k * c + (k - k_ - 1) * c * fd.d_dx(self.x) + c * fd.d2_dx2(self.x)
			B_t_T = self.discount[-1] / disc_t
			Vhold = V
			
			bounds = self.payoff.boundary(t, S, self, locals())
			if 'dirichlet' in bounds:
				dirichlet_lo, dirichlet_hi = bounds['dirichlet']
				dirichlet_lo = self.V_to_u(dirichlet_lo) if dirichlet_lo is not None else None
				dirichlet_hi = self.V_to_u(dirichlet_hi) if dirichlet_hi is not None else None
				bounds['dirichlet'] = dirichlet_lo, dirichlet_hi
			if 'neumann' in bounds:
				neumann_lo, neumann_hi = bounds['neumann']
				#du/dx = dV/dS du/dV dS/dx = u dV/dS
				if neumann_lo is not None:
					neumann_lo = u[0] * neumann_lo # tmp!
				if neumann_hi is not None:
					neumann_hi = u[-1] * neumann_hi # tmp!
				bounds['neumann'] = neumann_lo, neumann_hi
			
			if self.iteration:
				solver = {
					'jacobi': fd.solve_jacobi,
					'gauss-seidel': fd.solve_gauss_seidel,
				}[self.iteration]
				
				if method == 'crank-nicolson':
					rhs = (1 + L * dtau / 2)(u)
					A = (1 - L * dtau / 2)
				elif method == 'implicit':
					rhs = u
					A = 1 - L * dtau
				else:
					raise NotImplementedError, "iteration not defined for fd scheme %s" % method
				
				def project_sol(u):
					u = proj(t, S, u)
					fd.set_explicit_boundaries(u, ** bounds)
					return u
				u = solver(A, rhs, u, 1e-10, 100, project_sol)
				
			else:
				step = stepsmap[method]
				u = step(L, 0, dtau, u, ** bounds)
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
	
	def boundary(self, t, S, pde, pdevars):
		raise NotImplementedError

class AmericanPayoff(VanillaPayoff):
	
	def early_ex(self, t, S, Vhold):
		V = self.expire(t, S)
		V[V < Vhold] = Vhold
		return V

class EuropeanPayoff(VanillaPayoff):
	
	def early_ex(self, t, S, Vhold):
		return Vhold

	def boundary(self, t, S, pde, pdevars):
		B_t_T = pdevars['B_t_T']
		if self.CP == 'C':
			return dict(
				dirichlet = (0., S[-1] - self.K * B_t_T)
			)
		else:
			return dict(
				dirichlet = (self.K * B_t_T - S[0], 0.)
			)

if __name__ == '__main__':
	
	def l1(x):
		return numpy.mean(numpy.abs(x))
	
	def l2(x):
		return numpy.sqrt(numpy.mean(x * x))
	
	def linf(x):
		return numpy.max(numpy.abs(x))
	
	S0 = 100.
	K = 100.
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
	
	nexpl = 20
	#method = [ 'explicit' ] * nexpl + [ 'crank-nicolson' ] * (Nt - 1 - nexpl)
	#method = 'explicit'
	method = 'implicit'
	errmax = .6e-2
	#iteration = None
	iteration = 'jacobi'
	#iteration = 'gauss-seidel'
	
	for CP in "CP":
		payoff = EuropeanPayoff(K, CP)
		pde = BSPde(payoff, S, t, r, b, [], sigma, method, iteration = iteration)
		res = pde.solve(outputs = [ "full_grid" ])
		V = res['price']
		Vt = numpy.array(res['full_grid'])
		
		from bs import black_scholes_1973
		Vref = black_scholes_1973(T, S, sigma, r, b, K, CP)
		
		err = l2(V - Vref)
		print err
		assert(err < errmax)
		
