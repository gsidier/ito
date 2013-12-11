import numpy
from numpy import log, exp, sqrt
from scipy.stats import norm

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
		V = exp(- r * T) * (K * N(d1) - F * N(d2))
	else:
		raise ValueError, "Bad value for 'CP' parameter: " + CP
	if greeks:
		res['price'] = V
	else:
		res = V
	if 'delta' in greeks:
		if CP == 'C':
			delta = exp(- b * T) * N(d1)
		else:
			delta = - exp(- b * T) * N(- d1)
		res['delta'] = delta
	if 'gamma' in greeks:
		gamma = exp(- b * T) * phi(d1) / (S * sigma_sqrtT)
		res['gamma'] = gamma
	if 'vega' in greeks:
		vega = S * numpy.exp(- b * T) * phi(d1) * numpy.sqrt(T)
		res['vega'] = vega
	return res

