import numpy

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
	def __init__(self, l, d, u):
		n = len(d)
		u = numpy.insert(u, 0, 0)
		l = numpy.insert(l, len(l), 0)
		self.bands = numpy.array(map(list, [u, d, l]))
	
	def __call__(self, y):
		z = self.bands * y
		res = z[1, :]
		res[:-1] += z[0, 1:]
		res[1:] += z[2, :-1]
		return res

def noboundary(l, d, u):
	l = numpy.hstack([l, [0]])
	d = numpy.hstack([[0], d, [0]])
	u = numpy.hstack([[0], u])
	return FinDiffOp(l, d, u)

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

