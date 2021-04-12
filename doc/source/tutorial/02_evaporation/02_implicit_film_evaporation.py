# Copyright (c) 2019-2020,  Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: astatt / Everyone is free to add additional tutorials

import numpy as np
import sys
import hoomd
from hoomd import md
from hoomd import data
try:
    from hoomd import azplugins
except ImportError:
    import azplugins

from scipy.spatial.distance import cdist

def init_mixture(system,snapshot,rho_B,rho_A,height,s_B,s_A,kT):
    Lx = system.box.Lx
    Ly = system.box.Ly
    Lz = system.box.Lz
    N_B = int(rho_B*Lx*Ly*height)
    N_A = int(rho_A*Lx*Ly*height)
    s_AB = 0.5*(s_B+s_A)

    snapshot.particles.resize(N_A+N_B)

    lx = np.arange(-Lx/2.+s_A/2.,+Lx/2.,s_A)
    ly = np.arange(-Ly/2.+s_A/2.,+Ly/2.,s_A)
    lz = np.arange(-Lz/2.+s_AB,-Lz/2.+s_AB+height,s_A)
    positions_Square_lattice_A = np.asarray(np.meshgrid(lx, ly, lz)).reshape(3,-1).transpose()
    np.random.shuffle(positions_Square_lattice_A)
    pos_A = positions_Square_lattice_A[:N_A]

    lx = np.arange(-Lx/2.+s_A/2.,+Lx/2.,s_B)
    ly = np.arange(-Ly/2.+s_A/2.,+Ly/2.,s_B)
    lz = np.arange(-Lz/2.+s_AB,-Lz/2.+s_AB+height,s_B)
    positions_Square_lattice_B = np.asarray(np.meshgrid(lx, ly, lz)).reshape(3,-1).transpose()

    dist = cdist(positions_Square_lattice_B,pos_A)
    eps=0.5
    cut = np.any(dist<0.5*(s_A+s_B),axis=1)
    positions_Square_lattice_B= positions_Square_lattice_B[~cut]
    np.random.shuffle(positions_Square_lattice_B)
    pos_B = positions_Square_lattice_B[:N_B]
    pos = np.vstack((pos_A,pos_B))
    mass_ratio = s_A**3/s_B**3
    snapshot.particles.typeid[:]= np.hstack((np.zeros(N_A),np.ones(N_B)))
    snapshot.particles.mass[:]= np.hstack((np.ones(N_A)*mass_ratio,np.ones(N_B)))
    snapshot.particles.position[:] = pos
    snapshot.particles.velocity[:] = np.random.normal(0, np.sqrt(kT), (N_A+N_B,3))
    snapshot.particles.velocity[:] -= np.average(snapshot.particles.velocity,axis=0)

    return snapshot

L = 25
Lz = 55.0
height = 50
kT = 1.0

rho_A = 0.005
s_A  = 4.0
rho_B = 0.1
s_B  = 1.0

hoomd.context.initialize()
hoomd.context.SimulationContext()

snapshot = hoomd.data.make_snapshot(N=0,box=data.boxdim(Lx=L,Ly=L,Lz=Lz+1e-3),particle_types=['A','B'])
system = hoomd.init.read_snapshot(snapshot)

snapshot_init = init_mixture(system,snapshot,rho_B,rho_A,height,s_B,s_A,kT)
system.restore_snapshot(snapshot_init)

s_AB = 0.5*(s_B+s_A)
nl = hoomd.md.nlist.cell()
lj = hoomd.md.pair.lj(nlist=nl,r_cut =3.0*s_A,name='n')
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=s_A, r_cut=2**(1/6.)*s_A)
lj.pair_coeff.set('B', 'A', epsilon=1.0, sigma=s_AB, r_cut=2**(1/6.)*s_AB)
lj.pair_coeff.set('B', 'B', epsilon=1.0, sigma=s_B, r_cut=2**(1/6.)*s_B)

lower_wall=hoomd.md.wall.group()
lower_wall.add_plane((0,0,-Lz/2.), (0,0,1))
lj_wall_lo=azplugins.wall.lj93(lower_wall, r_cut=3.0*s_AB,name='wall')
lj_wall_lo.force_coeff.set(['A','B'], epsilon=2.5, sigma=s_AB, r_cut=3.0*s_AB)

upper_wall=hoomd.md.wall.group()
upper_wall.add_plane((0,0,Lz/2.), (0,0,-1))
lj_wall_up=azplugins.wall.lj93(upper_wall, r_cut=s_AB*(2/5.)**(1/6.))
lj_wall_up.force_coeff.set(['A','B'], epsilon=2.5, sigma=s_AB, r_cut=(2/5.)**(1/6.)*s_AB)

all = hoomd.group.all()

hoomd.dump.gsd(filename="tutorial_01_implicit_evaporation_trajectory.gsd",
               overwrite=True, period=5e3, group=all)

class measure_evaporation:
    def __init__(self, system, binsize):
        self.system   = system
        self.binsize  = binsize
        self.num_bins = np.round(self.system.box.Lz/self.binsize).astype(int)
        self.types    = np.asarray(self.system.particles.types)
        self.H_dens   = np.zeros((len(self.types),self.num_bins))
        self.counter  = 0
        self.range    = [-self.system.box.Lz/2.,self.system.box.Lz/2.]
        self.bin_vol  = float(self.system.box.Lz)/float(self.num_bins)*self.system.box.Lx*self.system.box.Ly

    def __call__(self, timestep):
        hoomd.util.quiet_status()
        snap = self.system.take_snapshot()
        hoomd.util.unquiet_status()
        pos = snap.particles.position
        # pick z coordinates
        pos_z = pos[:,2]
        for t in self.types:
            i = np.argwhere(self.types==t)[0][0]
            H_dens, edges = np.histogram(pos_z[np.where(snap.particles.typeid==i)],bins = self.num_bins,range=self.range)
            self.H_dens[i]  = H_dens
        centers  =  (edges[:-1] + edges[1:])/2
        temp = np.zeros(len(centers))
        # temperature histogram
        for i,c in enumerate(centers):
            slab_vel = snap.particles.velocity[np.abs(pos[:,2]-c)<self.binsize]
            l = len(slab_vel)
            if l>0:
                v_squared = slab_vel[:,0]**2 + slab_vel[:,1]**2 + slab_vel[:,2]**2
                T = 1/(3*l)*np.sum(v_squared)
            else:
                T=0
            temp[i]=T

        # normalize density
        to_save_H = self.H_dens/self.bin_vol

        res = np.vstack((np.asarray(centers),to_save_H,temp)).T
        np.savetxt('tutorial_evaporation_implicit_dens_%05d.hist'%hoomd.get_step(),res, header="z, density %s temp"%self.types)

        self.counter += 1

o = measure_evaporation(system, binsize=1.0)
analyze =  hoomd.analyze.callback(o, period=5e3)

hoomd.md.integrate.mode_standard(dt = 0.005)
langevin = hoomd.md.integrate.langevin(group=all, kT=kT, seed=457)

interf = hoomd.variant.linear_interp([[0,Lz/2.],[1e4,Lz/2.],[6e5,-Lz/2.+5]],zero=0)
evap = azplugins.evaporate.implicit(interface=interf)
k = 50.0
evap.force_coeff.set('B', k=k*(s_B/2.)**2, offset=0.0, g=k*(s_B/2.)**3, cutoff=s_B/2.)
evap.force_coeff.set('A', k=k*(s_A/2.)**2, offset=0.0, g=k*(s_A/2.)**3, cutoff=s_A/2.)

langevin.set_gamma('A', gamma=2.5*s_A/s_B)
langevin.set_gamma('B', gamma=2.5)

hoomd.run(6e5)
