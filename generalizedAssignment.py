#coding: utf-8
import sys
import math
import random
import collections
from gurobipy import *

def getInstance():

	tasks = [1, 2, 3]
	resources = [1, 2]

	c = [[0, 0, 0], [0, 9, 2], [0, 1, 2], [0, 3, 8]]
	a = [[0, 0, 0], [0, 6, 8], [0, 7, 5], [0, 9, 6]]
	b = [0, 13, 11]

	return (tasks, resources, c, a, b)

def solveRelax():

	(tasks, resources, c, a, b) = getInstance()

	model = Model('Generalized Assignment Problem')

	x = model.addVars(tasks, resources, vtype = GRB.BINARY, name = 'x')

	model.update()

	model.setObjective(quicksum(c[i][j] * x[i, j] for i in tasks for j in resources), GRB.MINIMIZE)

	model.addConstrs((quicksum(x[i, j] for j in resources) == 1 for i in tasks), name = 'r1')
	model.addConstrs((quicksum(a[i][j] * x[i, j] for i in tasks) <= b[j] for j in resources), name = 'r2')

	model = model.relax()

	model.optimize()

	print('objective function value = {}'.format(model.objVal))
	objValue = model.objVal

	model.printAttr('x')

	print('\n    Constraint          r')
	print('-------------------------')

	dualValue = []
	dualValue.append(0.0)

	constrNames = []
	for i in tasks:
		constrNames.append('r1[{}]'.format(i))

	for i in model.getConstrs():
		if i.ConstrName in constrNames:
			print('       {}     {:.6f}'.format(i.ConstrName, i.Pi))
			dualValue.append(i.Pi)

	model.reset()
	model.remove(model.getVars())
	model.remove(model.getConstrs())

	return (objValue, dualValue)

def solveLagrangianModel(u):

	(tasks, resources, c, a, b) = getInstance()

	model = Model('Generalized Assignment Problem')

	x = model.addVars(tasks, resources, vtype = GRB.BINARY, name = 'x')

	model.update()

	model.setObjective(quicksum((c[i][j] - u[i]) * x[i, j] for i in tasks for j in resources) + quicksum(u[i] for i in tasks), GRB.MINIMIZE)

	model.addConstrs((quicksum(a[i][j] * x[i, j] for i in tasks) <= b[j] for j in resources), name = 'r2')

	model.optimize()

	model.printAttr('x')

	solution = [[0 for j in range(len(resources) + 1)] for i in range(len(tasks) + 1)]

	for i in tasks:
		for j in resources:
			solution[i][j] = x[i, j].x

	objValue = model.objVal

	model.reset()
	model.remove(model.getVars())
	model.remove(model.getConstrs())

	return (objValue, solution)	

def stepsize(Lu, xu, optL, teta):

	(tasks, resources, c, a, b) = getInstance()

	y = [1 for i in range(len(tasks) + 1)]
	for i in tasks:
		for j in resources:
			y[i] -= xu[i][j]

	t = (teta * (optL - Lu)) / sum(math.pow(y[i], 2) for i in tasks)

	return (y, t)

def transition(Lu, xu, optL, teta, u):

	(tasks, resources, c, a, b) = getInstance()
	(y, t) = stepsize(Lu, xu, optL, teta)
	
	for i in tasks:
		u[i] = max(0, u[i] + t * y[i])

	return (u)
		
if __name__ == '__main__':

	history = []

	optL = 19
	teta = 1.0
	e = 0.00001

	step = 0
	stepMemory = step

	print('\n\n------------------------ LINEAR SOLUTION ------------------------\n')

	(memory, u) = solveRelax()

	print('\n\n---------------------- LAGRANGIAN SOLUTION ----------------------\n')

	while True:

		(Lu, xu) = solveLagrangianModel(u)

		history.append({'u[{}]'.format(step): u[1:len(u)], 'fo(u[{}])'.format(step): Lu, 'x(u[{}])'.format(step): xu, 'type': 'step up__'})

		step += 1

		if memory <= Lu:
			if Lu - memory > e:
				teta = float(teta * 2)
				stepMemory = step - 1
				memory_u = u
				memory = Lu
				u = transition(Lu, xu, optL, teta, u)
			else:
				break
		else:
			u[1:len(u)] = history[stepMemory]['u[{}]'.format(stepMemory)]
			Lu = history[stepMemory]['fo(u[{}])'.format(stepMemory)]
			xu = history[stepMemory]['x(u[{}])'.format(stepMemory)]
			history.append({'u[{}]'.format(step): u[1:len(u)], 'fo(u[{}])'.format(step): Lu, 'x(u[{}])'.format(step): xu, 'type': 'step down'})
			teta = float(teta / 2)
			u = transition(Lu, xu, optL, teta, u)
			step += 1

	for i in history:
		i = collections.OrderedDict(sorted(i.items()))
		print(i)
