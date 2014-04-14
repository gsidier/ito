import numpy
from numpy import log, exp, sqrt
from scipy.stats import norm

_VOMMA = 'vomma volga dvegadvol'.split()
_VANNA = 'vanna dvegadspot ddeltadvol'.split()
_ZOMMA = 'zomma dgammadvol'.split()

def black_scholes_1973(T, S, sigma, r, b, K, CP, greeks = []):
	"""
	greeks: a list of optional outputs:
		delta
		gamma
		forward
		d1
		d2
	
	output: a dict containing the key 'price' and any greeks specified in input.
	"""
	res = { }
	greeks = set(greeks)
	F = S * exp((r - b) * T)
	if 'forward' in greeks:
		res['forward'] = F
	sigma_sqrtT = sigma * sqrt(T)
	sigma2_T = sigma_sqrtT * sigma_sqrtT
	d0 = log(F / K) / sigma_sqrtT
	d1 = d0 + .5 * sigma_sqrtT
	d2 = d0 - .5 * sigma_sqrtT
	if 'd1' in greeks:
		res['d1'] = d1
	if 'd2' in greeks:
		res['d2'] = d2
	N = norm.cdf
	phi = norm.pdf
	CP = str(CP)[0].upper()
	if CP == 'C':
		V = exp(- r * T) * (F * N(d1) - K * N(d2))
	elif CP == 'P':
		V = exp(- r * T) * (K * N(- d2) - F * N(- d1))
	else:
		raise ValueError, "Bad value for 'CP' parameter: " + CP
	if greeks:
		res['price'] = V
	else:
		res = V
	if 'delta' in greeks or 'theta' in greeks:
		if CP == 'C':
			delta = exp(- b * T) * N(d1)
		else:
			delta = - exp(- b * T) * N(- d1)
	if 'delta' in greeks:
		res['delta'] = delta
	if 'gamma' in greeks or 'theta' in greeks or any(g in greeks for g in _ZOMMA):
		gamma = exp(- b * T) * phi(d1) / (S * sigma_sqrtT)
	if 'gamma' in greeks:
		res['gamma'] = gamma
	if vega in greeks or any(g in greeks for g in _VOMMA) or any(g in greeks for g in _VANNA):
		vega = S * numpy.exp(- b * T) * phi(d1) * numpy.sqrt(T)
	if 'vega' in greeks:
		res['vega'] = vega
	if 'theta' in greeks:
		theta = r * V - (r - b) * delta * S - .5 * gamma * S * S * sigma * sigma
		res['theta'] = theta
	if any(g in greeks for g in _VOMMA):
		volga = vega * d1 * d2 / sigma
		for g in VOMMA:
			if g in greeks:
				res[g] = volga
	if any(g in greeks for g in _VANNA):
		vanna = vega / S * (1 - d1 / sigma_sqrtT)
		for g in _VANNA:
			if g in greeks:
				res[g] = vanna
	if any(g in greeks for g in _ZOMMA):
		zomma = gamma * (d1 * d2 - 1) / sigma
		for g in _ZOMMA:
			if g in greeks:
				res[g] = zomma
	return res

