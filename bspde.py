from __future__ import print_function
import __init__
import numpy as np
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
		raise NotImplementedError()
	
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
		raise NotImplementedError()
	
	def early_ex(self, t, S, Vhold):
		"""
		Input:
			t: single time
			S: vector of underlyer prices at expiry
			Vhold: vector of option price if not exercised
		
		Return:
			V: vector of option prices
		"""
		raise NotImplementedError()

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
	
	Cash dividends:
	
		X = X0 exp(x)
		Dt = sum[ti > = t] B(t, ti) di
		St = Xt + Dt
		dX / X = (r - b) dt + sigma dW
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
		self.r = r + np.zeros(self.Nt - 1)
		self.b = b + np.zeros(self.Nt - 1)
		self.divs = divs
		self.sigma = sigma + np.zeros(self.Nt - 1)
		#if isinstance(method, (str, unicode)):
		if isinstance(method, str):
			self.method = [ method ] * (self.Nt - 1)
		else:
			self.method = method
		self.iteration = iteration
		self.discount = np.insert(np.cumprod([np.exp(- rt * dt) for (rt, dt) in zip(self.r[:len(self.t) - 1], self.t[1:] - self.t[:-1]) ]
				), 0, 1.) ## np.exp(- r * t)
		
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
		sumdivs_t = np.cumsum(divs_t[::-1])[::-1]
		self.D = sumdivs_t / self.discount
		
		self.X0 = S[int(len(S) / 2)] - self.D[0]
		t0 = t[0]
		self.x = self.S_to_x(0, S)
		self.T = t[-1]
		dtau = (t[1:] - t[:-1]) * self.sigma * self.sigma / 2
		self.tau = np.append(np.cumsum(dtau[::-1])[::-1], 0)
	
	def S_to_x(self, i, S):
		return self.X_to_x(S - self.D[i]) 
	
	def x_to_S(self, i, x):
		return self.x_to_X(x) + self.D[i]
	
	def X_to_x(self, X):
		return np.log(X / self.X0)
	
	def x_to_X(self, x):
		return self.X0 * np.exp(x)
	
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
			sigma = sigma + np.zeros(self.Nt - 1)
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
		St = [ ]
		ut = [ ]
		
		for (i, t, tau, dtau, c, k, k_, disc_t, method) in reversed(list(zip(range(self.Nt - 1), self.t[:-1], self.tau[:-1], self.tau[:-1] - self.tau[1:], C, K, K_, self.discount, self.method))):
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
				# du = du/dV dV/dS dS/dx dx = exp(x) dV/dS dx
				if neumann_lo is not None:
					neumann_lo *= .5 * (np.exp(self.x[0]) + np.exp(self.x[1])) * (self.x[1] - self.x[0])
				if neumann_hi is not None:
					neumann_hi *= .5 * (np.exp(self.x[-1]) + np.exp(self.x[-2])) * (self.x[-1] - self.x[-2])
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
					raise NotImplementedError("iteration not defined for fd scheme %s" % method)
				
				u = solver(A, rhs, u, 1e-10, 100, lambda u: proj(t, S, u), ** bounds)
				
			else:
				step = stepsmap[method]
				u = step(L, 0, dtau, u, ** bounds)
				u = proj(t, S, u)
			
			V = self.u_to_V(u)
			if 'full_grid' in outputs:
				Vt.append(V)
				St.append(S)
				ut.append(u)
		
		V = self.u_to_V(u)
		if 'full_grid' in outputs:
			V = np.array(Vt[::-1])
			self.Vt = Vt
			self.St = np.array(St[::-1])
			self.ut = np.array(ut[::-1])
		
		res['price'] = V
		
		
		if 'delta' in outputs or 'gamma' in outputs:
			d_dx = fd.d_dx(self.x)
			du_dx = d_dx(u)
		
		if 'delta' in outputs or 'vanna' in outputs:
			# x = log(S/X0): dx/dS = 1/S
			# dV/dS = dV/du du/dx dx/dS = X0/S du/dx
			delta = du_dx * self.X0 / self.S
			delta[0] = delta[1]
			delta[-1] = delta[-2]
			if 'delta' in outputs:
				res['delta'] = delta
		
		if 'gamma' in outputs:
			d2_dx2 = fd.d2_dx2(self.x)
			d2u_dx2 = d2_dx2(u)
			# d2V/dS2 = d(X0/S du/dx)/dS = -X0/S**2 du/dx + X0/S d2u/dx2 dx/dS = X0/S**2(d2u/dx2 - du/dx)
			gamma = self.X0 / (self.S * self.S) * (d2u_dx2 - du_dx)
			gamma[0] = gamma[-1] = 0
			res['gamma'] = gamma
		
		if any(volgreek in outputs for volgreek in "vega volga vanna".split()):
			outputs1 = ['price']
			if 'vanna' in outputs:
				outputs1.append('delta')
			dsigma = sigma[0] * 1e-2
			outPlus = self.solve(sigma=sigma+dsigma, outputs=outputs1)
			outMinus = self.solve(sigma=sigma-dsigma, outputs=outputs1)
			Vplus = outPlus['price']
			Vminus = outMinus['price']
			if 'vega' in outputs:
				res['vega'] = (Vplus - Vminus) / (2 * dsigma)
			if 'volga' in outputs:
				res['volga'] = (Vplus - 2*V + Vminus) / (dsigma*dsigma)
			if 'vanna' in outputs:
				deltaPlus = outPlus['delta']
				deltaMinus = outMinus['delta']
				res['vanna'] = (deltaPlus - deltaMinus) / (2*dsigma)
		
		if outputs:
			return res
		else:
			return V

class VanillaPayoff(object):
	
	def __init__(self, K, CP):
		self.K = K
		self.CP = str(CP)[0].upper()
		if self.CP not in "CP":
			raise ValueError("Unrecognized call/put flag: " + CP)
		
	def expire(self, t, S):
		if self.CP == 'C':
			V = (S - self.K)
		else:
			V = (self.K - S)
		V[V <= 0] = 0
		return V
	
	def boundary(self, t, S, pde, pdevars):
		raise NotImplementedError()

class AmericanPayoff(VanillaPayoff):
	
	def early_ex(self, t, S, Vhold):
		V = self.expire(t, S)
		hold = V < Vhold
		V[hold] = Vhold[hold]
		return V
	
	def boundary(self, t, S, pde, pdevars):
		B_t_T = pdevars['B_t_T']
		if self.CP == 'C':
			return dict(
				neumann = (0, 1.)
			)
		else:
			return dict(
				neumann = (-1., 0)
			)

class EuropeanPayoff(VanillaPayoff):
	
	def early_ex(self, t, S, Vhold):
		return Vhold

	def boundary(self, t, S, pde, pdevars):
		B_t_T = pdevars['B_t_T']
		if self.CP == 'C':
			return dict(
				#dirichlet = (0., S[-1] - self.K * B_t_T)
				neumann = (0, 1.)
			)
		else:
			return dict(
				#dirichlet = (self.K * B_t_T - S[0], 0.)
				neumann = (-1., 0)
			)

def genPriceGrid(S0, sigma, Texp, divs, r, nstd, Nx):
	sns = np.linspace(-nstd, +nstd, Nx)
	x = sns * sigma * np.sqrt(Texp)
	D = sum(np.exp(- r * ti) * di for (ti, di) in divs)
	X0 = S0 - D
	X = X0 * np.exp(x)
	S = X + D
	return S

def genTimeGrid(Texp, Nt):
	return np.linspace(0, Texp, Nt)

if __name__ == '__main__':
	
	def l1(x):
		return np.mean(np.abs(x))
	
	def l2(x):
		return np.sqrt(np.mean(x * x))
	
	def linf(x):
		return np.max(np.abs(x))
	
	X0 = 100.
	K = 100.
	Nx = 101
	Nt = 101
	sns = np.linspace(-6, +6, Nx)
	sigma = .3
	T = .25
	t = np.linspace(0, T, Nt)
	x = sns * sigma * np.sqrt(T)
	X = X0 * np.exp(x)
	r = 0.02
	b = 0
	divs = [(1. / 12., 10.)]
	D = sum(np.exp(- r * ti) * di for (ti, di) in divs)
	S = X + D
	
	nexpl = 20
	errmax = .6e-2
	
	settings = [ ]
	settings.append((None, [ 'explicit' ] * nexpl + [ 'crank-nicolson' ] * (Nt - 1 - nexpl)))
	settings.append(("jacobi", "implicit"))
	settings.append(("jacobi", "crank-nicolson"))
	settings.append(("gauss-seidel", "implicit"))
	settings.append(("gauss-seidel", "crank-nicolson"))
	
	full_grid = 1
	
	for iteration, method in settings:
		for CP in "CP":
			payoff = EuropeanPayoff(K, CP)
			pde = BSPde(payoff, S, t, r, b, divs, sigma, method, iteration = iteration)
			if full_grid:
				res = pde.solve(outputs = [ 'full_grid' ])
				Vt = res['price']
				V = Vt[0, :]
			else:
				res = pde.solve(outputs = [ "delta", "gamma" ])
				V = res['price']
				delta = res['delta']
				gamma = res['gamma']
			
			from bs import black_scholes_1973
			bs = black_scholes_1973(T, X, sigma, r, b, K, CP, greeks = ["delta", "gamma"])
			Vref = bs['price']
			deltaref = bs['delta']
			gammaref = bs['gamma']
			
			err = l2(V - Vref)
			print(err)
			assert(err < errmax)
			
