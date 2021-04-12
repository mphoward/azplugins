# Copyright (c) 2019-2020, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: astatt / Everyone is free to add additional tutorials

import numpy as np
import sys
import hoomd
from hoomd import md
from hoomd import data
try:
    from hoomd import azplugins # compiled as internal plugin
except ImportError:
    import azplugins # compiled externally


L = 10
kT = 0.772
rho = 0.849
sigma = 1.0

hoomd.context.initialize()
hoomd.context.SimulationContext()

snapshot = hoomd.data.make_snapshot(N=0,box=data.boxdim(L=L),particle_types=['A'])
system = hoomd.init.read_snapshot(snapshot)

N = int(rho*system.box.get_volume())
snapshot.particles.resize(N)

l = np.arange(-L/2.,+L/2.,sigma)
positions_square_lattice = np.asarray(np.meshgrid(l, l, l)).reshape(3,-1).transpose()
np.random.shuffle(positions_square_lattice)

snapshot.particles.position[:] = positions_square_lattice[:N]
snapshot.particles.velocity[:] = np.random.normal(0, np.sqrt(kT), (N,3))
snapshot.particles.velocity[:] -= np.average(snapshot.particles.velocity,axis=0)

system.restore_snapshot(snapshot)

nl = hoomd.md.nlist.cell()
lj = hoomd.md.pair.lj(r_cut=3.0, nlist=nl)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=sigma)

all = hoomd.group.all()


hoomd.md.integrate.mode_standard(dt = 0.005)
langevin = hoomd.md.integrate.langevin(group=all, kT=kT, seed=457)
hoomd.run(1e4)
langevin.disable()

vel_zx = azplugins.flow.FlowProfiler(system=system, bin_axis=2, flow_axis=0, bins=100, range=(-L/2,L/2), area=L**2)
analyze = hoomd.analyze.callback(vel_zx, period=1e2)

log = hoomd.analyze.log(filename="tutorial_reverse_perturbation_1_LJ.log",
                        quantities=['rp_momentum'],
                        period=1e2,overwrite=True)
hoomd.dump.gsd(filename="tutorial_reverse_perturbation_1_LJ_trajectory.gsd",
               overwrite=True, period=1e2, group=all)

nve = hoomd.md.integrate.nve(group = all)
f = azplugins.flow.reverse_perturbation(group=all,width=1,Nswap=1,period=10,target_momentum=0.5)
hoomd.run(5e5)
np.savetxt('tutorial_reverse_perturbation_1_LJ_vx.hist', np.column_stack((vel_zx.centers, vel_zx.velocity)))
