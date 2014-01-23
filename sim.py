import numpy
from bisect import bisect_left

class Sim(object):
	
	def __init__(self, t, P, cash, ** instruments):
		self.t = t
		self.P = P
		self.cash = cash
		self.instruments = instruments
		# checks:
		Nt, Nx, Nx_ = numpy.shape(self.P)
		assert(Nx == Nx_)
		# derived
		self.cdf = numpy.cumsum(self.P, 2)
		
	def random_transition(self, i, j, u):
		k = bisect.bisect_left(self.cdf[i, j, :], u)
		return k
	
	def random_scenario(self, initial):
		j = initial
		Nt = len(self.P) - 1
		rand = numpy.random.rand(Nt)
		states = [ ]
		for i, u in enumerate(rand):
			k = self.random_transition(i, j, u)
			states.append(k)
		return numpy.array(list(enumerate(states)))
	
	def sim(self, scenario, strategy):
		"""
		params:

			scenario: [ [ i, state ] ]
			strategy: time, market, pnl -> portfolio
		
		where
		
			portfolio: { instrument: allocation }
			market: { instrument: price }
		
		return: 
		
			[ pnl ]
		"""
		mktdata = dict((instr, price[scenario]) for (instr, price) in self.instruments.iteritems())
		pnl = [ ]
		pfolio = { }
		cash = 0
		for t, (i, j) in zip(self.t, enumerate(scenario)):
			pfolio = strategy(t, mktdata, pnl)
			if i > 0:
				cash = pnl[-1]
				cash *= self.cash[i, j] / self.cash[i_, j_]
				
				instrs = list(pfolio)
				allocs = list(pfolio[instr] for instr in instrs)
				mkt = list(mktdata[instr] for instr in instrs)
				prices_ = list(price[i_, j_] for price in mkt)
				prices = list(price[i, j] for price in mkt)
				
				cash -= sum(amt * price_ for (amt, price_) in zip(allocs, prices_)
				cash += sum(amt * price for (amt, price) in zip(allocs, prices)
			
			pnl.append(cash)
			i_ = i
			j_ = j
		return pnl

def stateful_strategy(strategy, initial = None):
	def strategy_(time, market_data, pnls):
		strategy_.state, portfolio = strategy(strategy_.state, time, market_data, pnls)
		return portfolio
	strategy_.state = initial
	return strategy_
