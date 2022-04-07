import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.signal import savgol_filter
from scipy.ndimage.filters import uniform_filter1d
import argparse

import seaborn as sns
sns.set_theme()
sns.set_style("whitegrid")

import matplotlib.colors as mc
import colorsys

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], min(1, amount * c[1]), c[2])

def plotTrajectory( root, obstacleId, output ):
    data = np.loadtxt(root+"/velocity_{}.dat".format(obstacleId), skiprows=1)

    fig, ax = plt.subplots(1, 2,sharex=True,figsize=(6,3), dpi=100)

    ax[0].plot(data[:,0], data[:,2], color="C0")
    ax[0].plot(data[:,0], data[:,3], color="C1")
    ax[0].set_xlabel("time $t$")
    ax[0].set_ylabel("position")
    # plt.ylim([0.8,1.4])

    ax[1].plot(data[:,0], data[:,7], label='x-components', color="C0")
    ax[1].plot(data[:,0], data[:,8], label='y-components', color="C1")
    ax[1].set_xlabel("time $t$")
    ax[1].set_ylabel("velocity")
    # plt.ylim([-1,1])

    index = data[:,0] > 10
    print("Average velocity:", np.mean(data[index,7]))

    ax[1].legend(bbox_to_anchor=(-0.5,-0.2, 2, 1), loc="lower left", frameon=False, ncol=2)
    plt.tight_layout()
    if output != "":
        plt.savefig(output)
    else:
        plt.show()

def plotForces( root, obstacleId, output ):
    data = np.loadtxt(root+"/forceValues_{}.dat".format(obstacleId), skiprows=1)

    fig, ax = plt.subplots(1, 2,sharex=True,figsize=(6,3), dpi=100)

    # 1/2 * speed of obstacle^2 * fish length
    nonDimFactor = 0.15**2/2*0.2

    ax[0].plot(data[:,0], data[:,10]/nonDimFactor, label='thrust', color="C0")
    ax[0].set_xlabel("time $t$")
    ax[0].set_ylabel("thrust $T$")
    ax[1].plot(data[:,0], -data[:,11]/nonDimFactor, label='drag', color="C1")
    ax[1].set_xlabel("time $t$")
    ax[1].set_ylabel("drag $D$")
    # twinAx.plot(data[:,0], data[:,7], label='torque', color="C2")

    # plt.legend()
    plt.tight_layout()
    if output != "":
        plt.savefig(output)
    else:
        plt.show()

def plotEnergy( root, obstacleId, output ):
    data = np.loadtxt(root+"/powerValues_{}.dat".format(obstacleId), skiprows=1)

    fig, ax = plt.subplots(1, 3, sharex=True,figsize=(12,3), dpi=100)

    # 1/2 * speed of obstacle^2 * fish length^2
    nonDimFactor = 0.15**2/2*0.2**2

    ax[0].plot(data[:,0], data[:,1]/nonDimFactor, color="C0")
    ax[0].set_xlabel("time $t$")
    ax[0].set_ylabel("thrust power $P_T$")
    ax[1].plot(data[:,0], data[:,2]/nonDimFactor, color="C1")
    ax[1].set_xlabel("time $t$")
    ax[1].set_ylabel("drag power $P_D$")
    ax[2].plot(data[:,0], data[:,7]/nonDimFactor, color="C2")
    ax[2].set_xlabel("time $t$")
    ax[2].set_ylabel("deformation power $P_S$")
    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots(1, 1, sharex=True,figsize=(6,3), dpi=100)
    ax.plot(data[:,0], data[:,-1], label='efficiency', color="C3") #or -1
    ax.set_xlabel("time $t$")
    ax.set_ylabel("efficiency $\eta$")
    plt.tight_layout()
    if output != "":
        plt.savefig(output)
    else:
        plt.show()

def compareDisplacement():
    y0 = 0.5
    output = "displacement.png"
    dirs = ["/scratch/snx3000/pweber/korali/MAcolumnFish_efficiency/_testingResults/sample0000", "/scratch/snx3000/pweber/korali/MAcolumnFish_yDisplacementMinimization/_testingResults/sample0000"]

    fig, ax = plt.subplots(1, 1, sharex=True,figsize=(6,3), dpi=100)

    for j, root in enumerate(dirs):
        for obstacleId in range(1,4):
            data = np.loadtxt(root+"/velocity_{}.dat".format(obstacleId), skiprows=1)
            ax.plot(data[:,0], data[:,3]-y0, color=lighten_color("C{}".format(j*2),obstacleId*0.5), linewidth=2) #or -1
    ax.set_xlabel("time $t$")
    ax.set_ylabel("lateral displacement $\Delta y$")
    plt.tight_layout()
    if output != "":
        plt.savefig(output)
    else:
        plt.show()

def compareForces():
    output = "forces.png"
    dirs = ["/scratch/snx3000/pweber/korali/MAcolumnFish_efficiency/_testingResults/sample0000", "/scratch/snx3000/pweber/korali/MAcolumnFish_yDisplacementMinimization/_testingResults/sample0000"]

    fig, ax = plt.subplots(1, 1, sharex=True,figsize=(6,3), dpi=100)

    # 1/2 * speed of obstacle^2 * fish length
    nonDimFactor = 0.15**2/2*0.2

    for j, root in enumerate(dirs):
        for obstacleId in range(1,4):
            data = np.loadtxt(root+"/forceValues_{}.dat".format(obstacleId), skiprows=1)
            ax.plot(data[:,0], -data[:,11]/nonDimFactor, color=lighten_color("C{}".format(j*2),obstacleId*0.5), linewidth=2) #or -1
    ax.set_xlabel("time $t$")
    ax.set_ylabel("lateral displacement $\Delta y$")
    plt.tight_layout()
    if output != "":
        plt.savefig(output)
    else:
        plt.show()

def compareEfficiency():
    output = "efficiency.png"
    dirs = ["/scratch/snx3000/pweber/korali/MAcolumnFish_efficiency/_testingResults/sample0000", "/scratch/snx3000/pweber/korali/MAcolumnFish_yDisplacementMinimization/_testingResults/sample0000"]

    fig, ax = plt.subplots(1, 1, sharex=True,figsize=(6,3), dpi=100)

    for j, root in enumerate(dirs):
        for obstacleId in range(1,4):
            data = np.loadtxt(root+"/powerValues_{}.dat".format(obstacleId), skiprows=1)
            # smoothned = savgol_filter(data[:,-1], 51, 3)
            averaged = uniform_filter1d(data[:,-1], size=20000)
            ax.plot(data[:,0], averaged, label='efficiency', color=lighten_color("C{}".format(j*2),obstacleId*0.5), linewidth=2) #or -1
    ax.set_xlabel("time $t$")
    ax.set_ylabel("averaged efficiency $\eta$")
    plt.tight_layout()
    if output != "":
        plt.savefig(output)
    else:
        plt.show()


def plotEfficiency():
    rootControlled = "/scratch/snx3000/mchatzim/CUP2D/School_Control_09Fish/"
    rootUncontrolled = "/scratch/snx3000/mchatzim/CUP2D/School_NoControl_09Fish/"
    rootSingle = "/scratch/snx3000/mchatzim/CUP2D/single-fish/"

    row = 9

    powDataSingle = np.loadtxt(rootSingle+"powerValues_0.dat", skiprows=1)
    powDataSingleAverage = powDataSingle[:,row]
    powDataSingleSmoothned = savgol_filter(powDataSingleAverage, 51, 3)
    plt.plot(powDataSingle[:,0], powDataSingleSmoothned, linewidth=1, label='single fish')
    powDataSingleAveraged = uniform_filter1d(powDataSingleAverage, size=1000)
    plt.plot(powDataSingle[:,0], powDataSingleAveraged, "--", linewidth=2, label='single fish, average')

    numSwimmersUncontrolled = 9
    powDataUncontrolled = np.loadtxt(rootUncontrolled+"powerValues_0.dat", skiprows=1)
    powDataUncontrolledAverage = np.zeros(powDataUncontrolled[:,row].shape)
    for i in range(0,numSwimmersUncontrolled):
        powDataUncontrolled = np.loadtxt(rootUncontrolled+"powerValues_{}.dat".format(i), skiprows=1)
        powDataUncontrolledAverage = powDataUncontrolledAverage + powDataUncontrolled[:,row]
    powDataUncontrolledSmoothened = savgol_filter(powDataUncontrolledAverage, 51, 3)
    plt.plot(powDataUncontrolled[:,0], powDataUncontrolledSmoothened/numSwimmersUncontrolled, linewidth=1, label='9 fish, uncontrolled')
    powDataUncontrolledAveraged = uniform_filter1d(powDataUncontrolledAverage, size=1000)
    plt.plot(powDataUncontrolled[:,0], powDataUncontrolledAveraged/numSwimmersUncontrolled, "--", linewidth=2, label='9 fish, uncontrolled, average')

    powDataControlled = np.loadtxt(rootControlled+"powerValues_0.dat", skiprows=1)
    powDataAverage = np.zeros(powDataControlled[:,row].shape)
    for i in range(0,9):
        powDataControlled = np.loadtxt(rootControlled+"powerValues_{}.dat".format(i), skiprows=1)
        powDataAverage = powDataAverage + powDataControlled[:,row]
    powDataAverageSmoothened = savgol_filter(powDataAverage, 51, 3)
    plt.plot(powDataControlled[:,0], powDataAverageSmoothened/9, linewidth=1, label='9 fish, controlled')
    powDataAverageAveraged = uniform_filter1d(powDataAverage, size=1000)
    plt.plot(powDataControlled[:,0], powDataAverageAveraged/9, "--", linewidth=2, label='9 fish, controlled, average')

    plt.xlim([0,30])
    plt.legend()
    plt.show()

def animateCoM():
    path = "/scratch/snx3000/pweber/korali/hydrofoil_largeDomain.eval/_testingResults/sample000/"
    numFish = 1
    
    # Load data, set up figure, axis, and plot element we want to animate
    fig = plt.figure()
    fig.patch.set_alpha(0.)
    ax = plt.axes(xlim=(0.4, 2),ylim=( 0.6, 1.4 ) )
    ax.set_axis_off()
    lines = []
    data = []
    for i in range(numFish):
        data.append( np.loadtxt(path+"velocity_{}.dat".format(i), skiprows=1) )
        lines.append(ax.plot([], [], lw=2, color="C{}".format(i))[0])

    # initialization function: plot the background of each frame
    def init():
      for line in lines:
        line.set_data([], [])
      return lines

    # animation function.  This is called sequentially
    def animate(i):
      plt.title("Center of Mass, t={}".format(data[0][i,0]))
      for j in range(numFish):
        lines[j].set_data(data[j][:i,2], data[j][:i,3])
      return lines

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(data[0][:,0]), interval=1, blit=True)

    # save the animation as an mp4
    anim.save('CoM_animation_fish.mp4', fps=10)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='cup2D.dynamics',
        description='Plot the dynamical quantities of a CUP2D run.')
    parser.add_argument(
        '--dir',
        help='Path to result files, separated by space',
        type = str,
        default = "",
        required=False)
    parser.add_argument(
        '--output',
        help='Name of saved figure.',
        type = str,
        default = "",
        required=False)
    parser.add_argument(
        '--obstacleId',
        help='Number of obstacle to plot.',
        type = int,
        default = 1,
        required=False)
    args = parser.parse_args()


    plotTrajectory(args.dir, args.obstacleId, args.output)
    plotForces(args.dir, args.obstacleId, args.output)
    plotEnergy(args.dir, args.obstacleId, args.output)
    # compareDisplacement()
    # compareForces()
    # compareEfficiency()
    # animateCoM()
    # plotEfficiency()