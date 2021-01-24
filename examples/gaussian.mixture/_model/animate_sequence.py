#! /usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import json
import argparse
import numpy as np

from gaussian_mixture import *


parser = argparse.ArgumentParser()
parser.add_argument( '--dir', help='directory of result files', default='_korali_result', required=False)
args = parser.parse_args()

path = args.dir

configFile = path + '/gen00000000.json'
if (not os.path.isfile(configFile)):
    print(f'[Korali] Error: Did not find any results in the {path} folder...')
    exit(-1)

with open(configFile) as f: js = json.load(f)
configRunId = js['Run ID']

resultFiles = [ f for f in os.listdir(path)
               if os.path.isfile(os.path.join(path, f)) and f.startswith('gen') ]

resultFiles = sorted(resultFiles)

genList = {}
for file in resultFiles:
    with open( path + '/' + file) as f: genJs = json.load(f)
    solverRunId = genJs['Run ID']
    if (configRunId == solverRunId):
        curGen = genJs['Current Generation']
        genList[curGen] = genJs

data = genList[next(iter(genList))]['Problem']['Data']
data = np.array(data)


x1 = np.amin( data[:,0] )
x2 = np.amax( data[:,0] )
y1 = np.amin( data[:,1] )
y2 = np.amax( data[:,1] )
dx = x2-x1
dy = y2-y1

x1 = x1 - 0.1*dx
x2 = x2 + 0.1*dx
y1 = y1 - 0.1*dy
y2 = y2 + 0.1*dy

x = np.linspace(x1,x2,200)
y = np.linspace(y1,y2,200)
x,y=np.meshgrid(x,y)
p = np.dstack((x,y))


class UpdatePlot(object):
    def __init__(self, ax):
        self.ax = ax
        self.ax.axis('off')
        self.alpha = 0.7
        self.sc = self.ax.scatter( [], [], s=40, alpha=self.alpha )
        self.ax.set_xlim(x1,x2)
        self.ax.set_ylim(y1,y2)
        self.cntr = []

    def init(self):
        rnd = np.random.normal(0, 1, size=data.shape)
        self.sc.set_offsets(data)
        return self.sc,

    def __call__(self, i):

        if i == 0:
            return self.init()

        try:
            for c in self.cntr.collections:
                c.remove()
        except:
            pass

        self.sc.remove()

        weights = np.array( genList[i]['Problem']['Weights'] )
        means = np.array( genList[i]['Problem']['Means'] )
        covariances = np.array( genList[i]['Problem']['Covariances'] )
        covariances = np.reshape(covariances, (-1, 2, 2))

        gmo = gm(means,covariances,weights)

        labels = np.argmax( gmo.pdfs(data), axis=1)
        pp = np.reshape( p, (p.shape[0]*p.shape[1],p.shape[2] ) )
        z = gmo.pdf(pp)
        z = np.reshape( z, (p.shape[0],p.shape[1]) )

        self.cntr = ax.contour(x, y, z, levels=20)
        self.sc = self.ax.scatter( data[:,0], data[:,1], c=labels, s=40, alpha=self.alpha)

        return self.cntr, self.sc

fig, ax = plt.subplots(1, 1, figsize=(10,10))
up = UpdatePlot( ax )
anim = FuncAnimation( fig, up, frames=genList, init_func=up.init, interval=1, blit=False)
plt.show()

writer = PillowWriter(fps=25)
anim.save("gm_em.gif", writer=writer)
