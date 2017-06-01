import matplotlib.pyplot as plt
import matplotlib.cm as cm

def makeScatterPlots(it, zk, qk): #, z0, q0):
    for k in range(layers):
        if k != layers-1: continue
        print("iter: %06d, plot layer " % it)
        plt.figure(k+1)
        plt.title("After Transformation %d" % (k+1))
        plt.xlabel("zk[%d][:,0]" % (k+1))
        plt.ylabel("zk[%d][:,1]" % (k+1))
        #plt.contourf(zk[k+1][:,0].reshape(grid_shape), zk[k+1][:,1].reshape(grid_shape), qk[k+1].reshape(grid_shape))
        plt.scatter(zk[k+1][:,0], zk[k+1][:,1], qk[k+1].reshape(-1))
        plt.savefig("transf_it%06d_layer%d.png" % (it, k))
        plt.close()

def makeScatterPlot(x, y, z, title, xlab, ylab, file_name): #, z0, q0):
        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.scatter(x, y, z)
        plt.savefig(file_name)
        plt.close()

def saveScatteredImage(z, q, filename="prior.png", directory="results"):
        N = 10
        plt.figure(figsize=(8, 6))
        plt.scatter(z[:, 0], z[:, 1], q.reshape(-1), marker='.')
        axes = plt.gca()
        axes.set_xlim([-3, 3])
        axes.set_ylim([-3, 3])
        plt.title("Prior: z0 vs z1")
        plt.grid(True)
        print("z= ", z.shape, z)
        print("q= ", q.shape, q)
        plt.savefig(directory + "/" + filename)
