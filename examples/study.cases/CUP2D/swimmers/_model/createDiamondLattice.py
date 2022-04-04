import numpy as np
import matplotlib.pyplot as plt


def printInitialPosition():
	# define size of domain
	Lx = 4
	Ly = 4

	# margins at boundary
	marginLeft        = 0.6
	marginRight       = 1
	yMargin           = 0.5

	# define diamond parameters
	dx = 0.3
	dy = 0.2

	# compute number of fish per direction
	Nx = int(( Lx - marginLeft - marginRight ) / dx) + 1
	NyOdd = int(( Ly - 2*yMargin ) / dy) + 1
	NyEven = NyOdd-1

	# create initialPosition
	print("std::vector<std::vector<double>> initialPositions{{")

	x0     = marginLeft
	y0odd  = yMargin
	y0even = yMargin+dy/2
	N = 0
	plt.xlim([0,Lx])
	plt.ylim([0,Ly])
	for i in range(Nx):
		if i % 2 == 0:
			y0 = y0even
			Ny = NyEven
		else:
			y0 = y0odd
			Ny = NyOdd

		for j in range(Ny):
			x = x0+i*dx
			y = y0+j*dy
			# plt.plot(x,y,"D")

			if( (i == Nx-1) and (j == Ny-1) ):
				print("{","{:.2f}, {:.2f}".format( x, y ),"}")
			else:
				print("{","{:.2f}, {:.2f}".format( x, y ),"},")

			N = N+1
	print("}};")
	print("This are initial condition for {} fish".format(N))
	# plt.show()

# For Diamond like school we have M^2 fish, i.e. 
def createDiamond( N, dx, dy ):
	L = 0.2
	x0 = 0.6
	xvel = 0.07

	# define size of domain
	Lx = 4
	Ly = 2

	# center of domain
	centerY = Ly / 2

	# compute number of fish in center of number of planes
	numCenter = int(np.sqrt(N))
	numPlanes = 2*numCenter - 1

	# create file and write options
	f = open("launchSwarm_N{}_dx{}_dy{}.sh".format(N, dx, dy), "w")

	f.write("#!/bin/bash\n\
OPTIONS=\"-bpdx 8 -bpdy 4 -levelMax 8 -levelStart 4 -Rtol 0.1 -Ctol 0.01 -extent 4 -CFL 0.2 -poissonTol 1e-11 -poissonTolRel 0  -bMeanConstraint 1 -bAdaptChiGradient 0 -tdump 0 -nu 0.00004 -tend 100 -muteAll 1 -verbose 0\"\n")

	# write front fish
	f.write("OBJECTS=\"stefanfish L={:.2f} T=1.0 xpos={:.2f} ypos={:.2f} xvel={:.2f} bForced=1 \n".format(L, x0, centerY, xvel))

	numFishY = 2
	bReduce = False
	for i in range(1,numPlanes):
		x  = x0 + i*dx

		if numFishY == numCenter:
			bReduce = True

		# compute y0 depending on which plane
		if i%2 != 0:
			y0 = centerY-(1+(numFishY//2-1)*2)*dy
		else:

			y0 = centerY-(numFishY-1)*dy

		# place fish in plane
		for j in range(numFishY):
			y = y0 + j*2*dy
			f.write('         stefanfish L={:.2f} T=1.0 xpos={:.2f} ypos={:.2f} xvel={:.2f} bForced=1 \n'.format(L, x, y, xvel))

		# Adjust the number of fish in the plane
		if not bReduce:
			numFishY = numFishY + 1
		else:
			numFishY = numFishY - 1

	# close OBJECT variable and finish up file
	f.write('\"\n')
	f.write('source launchCommon.sh')

if __name__ == '__main__':
	Ns = [4, 9, 16, 25]
	# Ns = [9] 
	deltaX = [ 0.2, 0.3, 0.4, 0.5, 0.6 ]
	# deltaX = [ 0.3 ]
	deltaY = [ 0.05, 0.1, 0.15, 0.2 ]
	# deltaY = [ 0.1 ]
	for N in Ns:
		for dx in deltaX:
			for dy in deltaY:
				createDiamond( N, dx, dy )