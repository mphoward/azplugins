# Copyright (c) 2019-2020, Michael P. Howard
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

def init_mixture(system,snapshot,rho_S,rho_A,rho_B,height,s_S,s_A,s_B,kT):
    Lx = system.box.Lx
    Ly = system.box.Ly
    Lz = system.box.Lz

    N_S = int(rho_S*Lx*Ly*height)
    N_A = int(rho_A*Lx*Ly*height)
    N_B = int(rho_B*Lx*Ly*height)
    s_AB = 0.5*(s_A+s_B)
    s_AS = 0.5*(s_A+s_S)
    s_BS = 0.5*(s_B+s_S)

    snapshot.particles.resize(N_A+N_B+N_S)

    lx = np.arange(-Lx/2.+s_A/2,+Lx/2.,s_A)
    ly = np.arange(-Ly/2.+s_A/2,+Ly/2.,s_A)
    lz = np.arange(-Lz/2.+s_A,-Lz/2.+s_A+height,s_A)
    positions_square_lattice_A = np.asarray(np.meshgrid(lx, ly, lz)).reshape(3,-1).transpose()
    np.random.shuffle(positions_square_lattice_A)
    pos_A = positions_square_lattice_A[:N_A]

    lx = np.arange(-Lx/2.+s_A/2,+Lx/2.,s_B)
    ly = np.arange(-Ly/2.+s_A/2,+Ly/2.,s_B)
    lz = np.arange(-Lz/2.+s_S,-Lz/2.+s_A+height,s_B)
    positions_square_lattice_B = np.asarray(np.meshgrid(lx, ly, lz)).reshape(3,-1).transpose()

    dist = cdist(positions_square_lattice_B,pos_A)
    cut = np.any(dist<s_AB,axis=1)
    positions_square_lattice_B= positions_square_lattice_B[~cut]
    np.random.shuffle(positions_square_lattice_B)
    pos_B = positions_square_lattice_B[:N_B]

    lx = np.arange(-Lx/2.+s_A/2,+Lx/2.,s_S)
    ly = np.arange(-Ly/2.+s_A/2,+Ly/2.,s_S)
    lz = np.arange(-Lz/2.+s_S,-Lz/2.+s_A+height,s_S)
    positions_square_lattice_S = np.asarray(np.meshgrid(lx, ly, lz)).reshape(3,-1).transpose()

    dist = cdist(positions_square_lattice_S,pos_A)
    cut = np.any(dist<s_AS,axis=1)
    positions_square_lattice_S= positions_square_lattice_S[~cut]

    dist = cdist(positions_square_lattice_S,pos_B)
    cut = np.any(dist<s_BS,axis=1)
    positions_square_lattice_S= positions_square_lattice_S[~cut]

    np.random.shuffle(positions_square_lattice_S)
    pos_S = positions_square_lattice_S[:N_S]

    pos = np.vstack((pos_A,pos_B,pos_S))
    mass_ratio_A = s_A**3/s_S**3
    mass_ratio_B = s_B**3/s_S**3

    snapshot.particles.typeid[:]= np.hstack((np.zeros(N_A),np.ones(N_B),np.ones(N_S)*2))
    snapshot.particles.mass[:]= np.hstack((np.ones(N_A)*mass_ratio_A,np.ones(N_B)*mass_ratio_B,np.ones(N_S)))
    snapshot.particles.position[:] = pos
    snapshot.particles.velocity[:] = np.random.normal(0, np.sqrt(kT), (N_A+N_B+N_S,3))
    snapshot.particles.velocity[:] -= np.average(snapshot.particles.velocity,axis=0)

    return snapshot

L = 20
Lz = 60
height = 40
kT = 1.0
rho_S = 0.5
rho_A = 0.001
rho_B = 0.001
s_S = 1.0
s_A = 4.0
s_B = 2.0

hoomd.context.initialize()
hoomd.context.SimulationContext()

snapshot = hoomd.data.make_snapshot(N=0,box=data.boxdim(Lx=L,Ly=L,Lz=Lz),particle_types=['A','B','S','T','Z'])
system = hoomd.init.read_snapshot(snapshot)

snapshot_init = init_mixture(system,snapshot,rho_S,rho_A,rho_B,height,s_S,s_A,s_B,kT)

system.restore_snapshot(snapshot_init)

s_AB = 0.5*(s_A+s_B)
s_AS = 0.5*(s_A+s_S)
s_BS = 0.5*(s_B+s_S)

nl = hoomd.md.nlist.cell()
lj = hoomd.md.pair.lj(nlist=nl,r_cut =3.0*s_AS,name='n')
lj.set_params(mode="xplor")
lj.pair_coeff.set(['S','T'], ['S','T'], epsilon=1.0, sigma=s_S, r_cut=3.0*s_S,      r_on=2.5*s_S)
lj.pair_coeff.set(['S','T'], 'A', epsilon=1.0, sigma=s_AS,r_cut=3.0*s_AS,     r_on=2.5*s_AS)
lj.pair_coeff.set(['S','T'], 'B', epsilon=1.0, sigma=s_BS,r_cut=3.0*s_BS,     r_on=2.5*s_BS)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=s_A, r_cut=2**(1/6.)*s_A,r_on=2.5*s_A)
lj.pair_coeff.set('B', 'B', epsilon=1.0, sigma=s_B, r_cut=2**(1/6.)*s_B,r_on=2.5*s_B)
lj.pair_coeff.set('A', 'B', epsilon=1.0, sigma=s_AB, r_cut=2**(1/6.)*s_AB,r_on=2.5*s_AB)
lj.pair_coeff.set('Z', ['A','B','S','Z','T'], epsilon=0.0, sigma=0, r_cut=False,r_on=2.5*s_A)

lower_wall=hoomd.md.wall.group()
lower_wall.add_plane((0,0,-Lz/2.), (0,0,1))
lj_wall_lo=azplugins.wall.lj93(lower_wall, r_cut=3.0*s_A,name='wall')
lj_wall_lo.force_coeff.set('A', epsilon=2.0, sigma=s_A, r_cut=3.0*s_A)
lj_wall_lo.force_coeff.set('B', epsilon=2.0, sigma=s_B, r_cut=3.0*s_B)
lj_wall_lo.force_coeff.set(['S','T','Z'], epsilon=2.0, sigma=s_S,  r_cut=3.0*s_S)

upper_wall=hoomd.md.wall.group()
upper_wall.add_plane((0,0,Lz/2.), (0,0,-1))
lj_wall_up=azplugins.wall.lj93(upper_wall, r_cut=s_A*(2/5.)**(1/6.))
lj_wall_up.force_coeff.set('A', epsilon=2.0, sigma=s_A, r_cut=(2/5.)**(1/6.)*s_A)
lj_wall_up.force_coeff.set('B', epsilon=2.0, sigma=s_B, r_cut=(2/5.)**(1/6.)*s_B)
lj_wall_up.force_coeff.set(['S','T','Z'], epsilon=2.0, sigma=s_S,  r_cut=(2/5.)**(1/6.)*s_S)


all = hoomd.group.all()

hoomd.md.integrate.mode_standard(dt = 0.005)
langevin = hoomd.md.integrate.langevin(group=all, kT=kT, seed=457)

hoomd.dump.gsd(filename="tutorial_02_explicit_evaporation_trajectory.gsd",
               overwrite=True, period=1e3, group=all,dynamic=['attribute','property','momentum'])
hoomd.run(5e3)

azplugins.update.types(inside='T', outside='S', lo=-Lz/2., hi=-Lz/2.+2, period=1)
langevin.set_gamma('T', gamma=0.1)
langevin.set_gamma('S', gamma=0.0)
evap = azplugins.evaporate.particles(solvent='S', evaporated='Z', lo=Lz/2.-2, hi=Lz/2., seed=77, period=1)


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
        self.outfile = open('tutorial_evaporation_explicit.txt', 'w+')
        self.outfile.write("# timestep N_sol\n")

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
        np.savetxt('tutorial_evaporation_explicit_dens_%05d.hist'%hoomd.get_step(),res, header="z, density %s temp"%self.types)

        # count solvent particles S+T
        N_sol = len(pos[np.where(np.logical_or(snap.particles.typeid==2,snap.particles.typeid==3))])
        self.outfile.write("%d %f \n"%(hoomd.get_step(),N_sol))
        self.outfile.flush()

        self.counter += 1

o = measure_evaporation(system, binsize=1.0)
analyze =  hoomd.analyze.callback(o, period=5e3)

hoomd.run(1e6)
