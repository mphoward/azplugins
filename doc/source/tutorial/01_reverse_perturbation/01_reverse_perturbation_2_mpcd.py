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
    import hoomd.azplugins.mpcd
except ImportError:
    import azplugins # compiled externally
    import azplugins.mpcd

L = 10
kT = 1.0
rho_mpcd = 5
viscosity_mpcd = 3.955

hoomd.context.initialize()
hoomd.context.SimulationContext()

snapshot = hoomd.data.make_snapshot(N=0,box=data.boxdim(L=L))
system = hoomd.init.read_snapshot(snapshot)

N_mpcd = int(rho_mpcd*system.box.get_volume())
snap = hoomd.mpcd.data.make_snapshot(N=N_mpcd)

snap.particles.position[:] = np.random.uniform(-L/2.0, L/2.0, (N_mpcd,3))
snap.particles.velocity[:] = np.random.normal(0, np.sqrt(kT), (N_mpcd,3))
snap.particles.velocity[:] -= np.average(snap.particles.velocity,axis=0)

mpcd_sys = hoomd.mpcd.init.read_snapshot(snap)

hoomd.mpcd.integrator(dt=0.1)
mpcd_sys.sorter.set_period(25)
srd   = hoomd.mpcd.collide.srd(seed=512, period=1, angle=130., kT=kT)
bulk  = hoomd.mpcd.stream.bulk(period=1)

# equilibration
hoomd.run(1e5)

vel_zx = azplugins.flow.FlowProfiler(system=system, bin_axis=2, flow_axis=0, bins=100, range=(-L/2,L/2), area=L**2)
analyze = hoomd.analyze.callback(vel_zx, period=1e2)
log = hoomd.analyze.log(filename="tutorial_reverse_perturbation_2_mpcd.log",
                        quantities=['rp_momentum'],
                        period=1e2,overwrite=True)
# flow
f = azplugins.mpcd.reverse_perturbation(width=1,Nswap=1,period=1,target_momentum=0.5)

hoomd.run(1e6)

snap = mpcd_sys.take_snapshot()
pos  = snap.particles.position
vel  = snap.particles.velocity
np.save('tutorial_reverse_perturbation_2_mpcd_pos.npy',pos)
np.save('tutorial_reverse_perturbation_2_mpcd_vel.npy',vel)
np.savetxt('tutorial_reverse_perturbation_2_mpcd_vx.hist', np.column_stack((vel_zx.centers, vel_zx.velocity)))
