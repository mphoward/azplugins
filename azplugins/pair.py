# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.

import math

import hoomd
from hoomd import _hoomd
from hoomd.md import _md

from . import _azplugins

class ashbaugh(hoomd.md.pair.pair):
    R""" Ashbaugh-Hatch potential

    Args:
        r_cut (float): Default cutoff radius (in distance units).
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list
        name (str): Name of the force instance.

    :py:class:`ashbaugh` is a Lennard-Jones perturbation potential, implemented as described by
    `Ashbaugh and Hatch <http://dx.doi.org/10.1021/ja802124e>`_. The potential has a purely
    repulsive (Weeks-Chandler-Andersen) core, with a parameter :math:`\lambda` setting the
    strength of the attractive tail. When :math:`\lambda` is 0, the potential is purely repulsive.
    When :math:`\lambda` is 1, the potential is the standard Lennard-Jones potential
    (see :py:class:`hoomd.md.pair.lj` for details).

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V(r)  = & V_{\mathrm{LJ}}(r, \varepsilon, \sigma, \alpha) + (1-\lambda)\varepsilon & r < (2/\alpha)^{1/6}\sigma \\
              = & \lambda V_{\mathrm{LJ}}(r, \varepsilon, \sigma, \alpha) & (2/\alpha)^{1/6}\sigma \ge r < r_{\mathrm{cut}} \\
              = & 0 & r \ge r_{\mathrm{cut}}
        \end{eqnarray*}

    Here, :math:`V_{\mathrm{LJ}}(r,\varepsilon,\sigma,\alpha)` is the stanard Lennard-Jones potential with
    parameters :math:`\varepsilon`, :math:`\sigma`, and :math:`\alpha`. See :py:class:`hoomd.md.pair.pair`
    for details on how forces are calculated and the available energy shifting and smoothing modes.
    Use :py:meth:`pair_coeff.set <coeff.set>` to set potential coefficients.

    The following coefficients must be set per unique pair of particle types:

    - :math:`\varepsilon` - *epsilon* (in energy units)
    - :math:`\sigma` - *sigma* (in distance units)
    - :math:`\alpha` - *alpha* (unitless) - *optional*: defaults to 1.0
    - :math:`\lambda` - *lam* (unitless) - scale factor for attraction
    - :math:`r_{\mathrm{cut}}` - *r_cut* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command
    - :math:`r_{\mathrm{on}}`- *r_on* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command

    Example::

        nl = hoomd.md.nlist.cell()
        ash = azplugins.pair.ashbaugh(r_cut=3.0, nlist=nl)
        ash.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, lam=0.0)
        ash.pair_coeff.set(['A','B'], 'B', epsilon=2.0, sigma=1.0, lam=0.5, r_cut=3.0, r_on=2.0)

    """
    def __init__(self, r_cut, nlist, name=None):
        hoomd.util.print_status_line()

        # initialize the base class
        hoomd.md.pair.pair.__init__(self, r_cut, nlist, name)

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_class = _azplugins.PairPotentialAshbaugh
        else:
            self.cpp_class = _azplugins.PairPotentialAshbaughGPU
            self.nlist.cpp_nlist.setStorageMode(_md.NeighborList.storageMode.full)
        self.cpp_force = self.cpp_class(hoomd.context.current.system_definition, self.nlist.cpp_nlist, self.name)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficent options
        self.required_coeffs = ['epsilon', 'sigma', 'alpha', 'lam']
        self.pair_coeff.set_default_coeff('alpha', 1.0)

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon']
        sigma = coeff['sigma']
        lam = coeff['lam']
        alpha = coeff['alpha']

        lj1 = 4.0 * epsilon * math.pow(sigma, 12.0)
        lj2 = alpha * 4.0 * epsilon * math.pow(sigma, 6.0)
        rwcasq = math.pow(2.0/alpha, 1.0/3.0) * sigma**2
        wca_shift = epsilon * alpha**2 * (1. - lam)
        return _azplugins.make_ashbaugh_params(lj1, lj2, lam, rwcasq, wca_shift)

class colloid(hoomd.md.pair.pair):
    R""" Colloid pair potential

    Args:
        r_cut (float): Default cutoff radius (in distance units).
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list
        name (str): Name of the force instance.

    :py:class:`colloid` is an effective Lennard-Jones potential obtained by
    integrating the Lennard-Jones potential between a point and a sphere or a sphere and a
    sphere. The attractive part of the colloid-colloid pair potential was derived originally
    by Hamaker, and the full potential by `Everaers and Ejtehadi <http://doi.org/10.1103/PhysRevE.67.041710>`_.
    A discussion of the application of these potentials to colloidal suspensions can be found
    in `Grest et al. <http://dx.doi.org/10.1063/1.3578181>`_

    The pair potential has three different coupling styles between particle types:

    - ``slv-slv`` gives the Lennard-Jones potential for coupling between pointlike particles

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V(r)  = & \frac{\varepsilon}{36} \left[\left(\frac{\sigma}{r}\right)^{12}
                  - \left(\frac{\sigma}{r}\right)^6 \right] & r < r_{\mathrm{cut}} \\
              = & 0 & r \ge r_{\mathrm{cut}}
        \end{eqnarray*}

    - ``coll-slv`` gives the interaction between a pointlike particle and a colloid
    - ``coll-coll`` gives the interaction between two colloids

    Refer to the work by `Grest et al. <http://dx.doi.org/10.1063/1.3578181>`_ for the
    form of the colloid-solvent and colloid-colloid potentials, which are too cumbersome
    to report on here. See :py:class:`hoomd.md.pair.pair` for details on how forces are
    calculated and the available energy shifting and smoothing modes. Use
    :py:meth:`pair_coeff.set <coeff.set>` to set potential coefficients.

    .. warning::
        The ``coll-slv`` and ``coll-coll`` styles make use of the particle diameters to
        compute the interactions. In the ``coll-slv`` case, the identity of the colloid
        in the pair is inferred to be the larger of the two diameters. You must make
        sure you appropriately set the particle diameters in the particle data.

    The strength of all potentials is set by the Hamaker constant, represented here by the
    symbol :math:`\varepsilon` for consistency with other potentials. The other parameter
    :math:`\sigma` is the diameter of the particles that are integrated out (colloids are
    comprised of Lennard-Jones particles with parameter :math:`\sigma`).

    The following coefficients must be set per unique pair of particle types:

    - :math:`\varepsilon` - *epsilon* (in energy units) - Hamaker constant
    - :math:`\sigma` - *sigma* (in distance units) - Size of colloid constituent particles
      - *optional*: defaults to 1.0
    - ``style`` - ``slv-slv``, ``coll-slv``, or ``coll-coll`` - Style of pair interaction
    - :math:`r_{\mathrm{cut}}` - *r_cut* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command
    - :math:`r_{\mathrm{on}}`- *r_on* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command

    Example::

        nl = hoomd.md.nlist.cell()
        coll = azplugins.pair.colloid(r_cut=3.0, nlist=nl)
        # standard Lennard-Jones for solvent-solvent
        coll.pair_coeff.set('S', 'S', epsilon=144.0, sigma=1.0, style='slv-slv')
        # solvent-colloid
        coll.pair_coeff.set('S', 'C', epsilon=100.0, sigma=1.0, style='slv-coll', r_cut=9.0)
        # colloid-colloid
        coll.pair_coeff.set('C', 'C', epsilon=40.0, sigma=1.0, r_cut=10.581)

    """
    def __init__(self, r_cut, nlist=None, name=None):
        hoomd.util.print_status_line()

        # initialize the base class
        hoomd.md.pair.pair.__init__(self, r_cut, nlist, name)

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_class = _azplugins.PairPotentialColloid
        else:
            self.cpp_class = _azplugins.PairPotentialColloidGPU
            self.nlist.cpp_nlist.setStorageMode(_md.NeighborList.storageMode.full)
        self.cpp_force = self.cpp_class(hoomd.context.current.system_definition, self.nlist.cpp_nlist, self.name)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficent options
        self.required_coeffs = ['epsilon', 'sigma', 'style']
        self.pair_coeff.set_default_coeff('sigma', 1.0)

    ## Process the coefficients
    def process_coeff(self, coeff):
        epsilon = coeff['epsilon']
        sigma = coeff['sigma']
        style = coeff['style']

        if style == 'slv-slv':
            style = 0
        elif style == 'coll-slv':
            style = 1
        elif style == 'coll-coll':
            style = 2
        else:
            hoomd.context.msg.error('Unknown interaction style\n')
            raise RuntimeError('Unknown interaction style')

        return _hoomd.make_scalar4(epsilon, sigma**3, sigma**6, _hoomd.int_as_scalar(style))

class hertz(hoomd.md.pair.pair):
    R"""Hertz potential

    Args:
        r_cut (float): Default cutoff radius (in distance units).
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list
        name (str): Name of the force instance.

    :py:class:`hertz` is the Hertz potential:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V(r)  &= \varepsilon ( 1-\frac{ r }{ r_{\rm{cut}} } )^{5/2} ,& r < r_{\rm{cut}} \\
              &= 0,& r \ge r_{\rm{cut}}
        \end{eqnarray*}

    parameters :math:`\varepsilon`, the energy scale of the pair interaction.
    See :py:class:`hoomd.md.pair.pair` for details on how forces are calculated
    and the available energy shifting and smoothing modes. Use
    :py:meth:`pair_coeff.set <coeff.set>` to set potential coefficients.

    The following coefficients must be set per unique pair of particle types:

    - :math:`\varepsilon` - *epsilon* (in energy units)
    - :math:`r_{\mathrm{cut}}` - *r_cut* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command

    Example::

        nl = hoomd.md.nlist.cell()
        hertz = azplugins.pair.hertz(r_cut=3.0, nlist=nl)
        hertz.pair_coeff.set('A', 'A', epsilon=1.0)
        hertz.pair_coeff.set(['A','B'], 'B', epsilon=2.0)

    """
    def __init__(self, r_cut, nlist, name=None):
        hoomd.util.print_status_line()

        # initialize the base class
        hoomd.md.pair.pair.__init__(self, r_cut, nlist, name)

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_class = _azplugins.PairPotentialHertz
        else:
            self.cpp_class = _azplugins.PairPotentialHertzGPU
            self.nlist.cpp_nlist.setStorageMode(_md.NeighborList.storageMode.full)
        self.cpp_force = self.cpp_class(hoomd.context.current.system_definition, self.nlist.cpp_nlist, self.name)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficent options
        self.required_coeffs = ['epsilon']

    def process_coeff(self, coeff):
        return coeff['epsilon']

class lj124(hoomd.md.pair.pair):
    R""" LJ 12-4 potential

    Args:
        r_cut (float): Default cutoff radius (in distance units).
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list
        name (str): Name of the force instance.

    :py:class:`lj124` is a Lennard-Jones potential

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V(r)  = & \frac{3 \sqrt{3}}{2} \varepsilon \left(\left(\frac{\sigma}{r}\right)^{12} - \alpha \left(\frac{\sigma}{r}\right)^4\right) &, r < r_{\mathrm{cut}} \\
              = & 0 &, r \ge r_{\mathrm{cut}}
        \end{eqnarray*}

    parameters :math:`\varepsilon`, :math:`\sigma`, and :math:`\alpha`. See :py:class:`hoomd.md.pair.pair`
    for details on how forces are calculated and the available energy shifting and smoothing modes.
    Use :py:meth:`pair_coeff.set <coeff.set>` to set potential coefficients.

    The following coefficients must be set per unique pair of particle types:

    - :math:`\varepsilon` - *epsilon* (in energy units)
    - :math:`\sigma` - *sigma* (in distance units)
    - :math:`\alpha` - *alpha* (unitless) - *optional*: defaults to 1.0
    - :math:`r_{\mathrm{cut}}` - *r_cut* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command

    Example::

        nl = hoomd.md.nlist.cell()
        lj124 = azplugins.pair.lj124(r_cut=3.0, nlist=nl)
        lj124.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        lj124.pair_coeff.set(['A','B'], 'B', epsilon=2.0, sigma=1.0, alpha=0.5)

    """
    def __init__(self, r_cut, nlist, name=None):
        hoomd.util.print_status_line()

        # initialize the base class
        hoomd.md.pair.pair.__init__(self, r_cut, nlist, name)

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_class = _azplugins.PairPotentialLJ124
        else:
            self.cpp_class = _azplugins.PairPotentialLJ124GPU
            self.nlist.cpp_nlist.setStorageMode(_md.NeighborList.storageMode.full)
        self.cpp_force = self.cpp_class(hoomd.context.current.system_definition, self.nlist.cpp_nlist, self.name)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficent options
        self.required_coeffs = ['epsilon', 'sigma', 'alpha']
        self.pair_coeff.set_default_coeff('alpha', 1.0)

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon']
        sigma = coeff['sigma']
        alpha = coeff['alpha']

        lj1 = 1.5 * math.sqrt(3.0) * epsilon * math.pow(sigma, 12.0)
        lj2 = alpha * 1.5 * math.sqrt(3.0) * epsilon * math.pow(sigma, 4.0)

        return _hoomd.make_scalar2(lj1, lj2)

class ashbaugh24(hoomd.md.pair.pair):
    R""" generalized Ashbaugh-Hatch 48-24 potential

    Args:
        r_cut (float): Default cutoff radius (in distance units).
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list
        name (str): Name of the force instance.

    :py:class:`ashbaugh24` is a Lennard-Jones 48-24 potential, implemented as described by
    `L. Rovigatti, B. Capone, C. Likos, Nanoscale, 8 (2016) <http://dx.doi.org/10.1039/C5NR04661K>`_.
    The potential has a purely repulsive (Weeks-Chandler-Andersen like) core, with a
    parameter :math:`\lambda` setting the strength of the attractive tail.
    When :math:`\lambda` is 0, the potential is purely repulsive.
    When :math:`\lambda` is 1, the potential a generalized Lennard-Jones potential:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V(r)  = & V_{\mathrm{LJ,48-24}}(r, \varepsilon, \sigma) + (1-\lambda)\varepsilon & r < (2)^{1/24}\sigma \\
              = & \lambda V_{\mathrm{LJ,48-24}}(r, \varepsilon, \sigma) & (2)^{1/24}\sigma \ge r < r_{\mathrm{cut}} \\
              = & 0 & r \ge r_{\mathrm{cut}}
        \end{eqnarray*}

    Here, :math:`V_{\mathrm{LJ,48-24}}(r,\varepsilon,\sigma)` is a Lennard-Jones potential with
    parameters :math:`\varepsilon`, and :math:`\sigma`:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{LJ,48-24}}  = 4 \varepsilon \left(\left(\frac{\sigma}{r}\right)^{48} - \left(\frac{\sigma}{r}\right)^{24}\right)
        \end{eqnarray*}

    See :py:class:`hoomd.md.pair.pair`
    for details on how forces are calculated and the available energy shifting and smoothing modes.
    Use :py:meth:`pair_coeff.set <coeff.set>` to set potential coefficients.

    The following coefficients must be set per unique pair of particle types:

    - :math:`\varepsilon` - *epsilon* (in energy units)
    - :math:`\sigma` - *sigma* (in distance units)
    - :math:`\lambda` - *lam* (unitless) - scale factor for attraction
    - :math:`r_{\mathrm{cut}}` - *r_cut* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command
    - :math:`r_{\mathrm{on}}`- *r_on* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command

    Example::

        nl = hoomd.md.nlist.cell()
        lj = azplugins.pair.ashbaugh24(r_cut=3.0, nlist=nl)
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, lam=0.0)
        lj.pair_coeff.set(['A','B'], 'B', epsilon=1.0, sigma=1.0, lam=0.5, r_cut=3.0, r_on=2.5)

    """
    def __init__(self, r_cut, nlist, name=None):
        hoomd.util.print_status_line()

        # initialize the base class
        hoomd.md.pair.pair.__init__(self, r_cut, nlist, name)

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_class = _azplugins.PairPotentialAshbaugh24
        else:
            self.cpp_class = _azplugins.PairPotentialAshbaugh24GPU
            self.nlist.cpp_nlist.setStorageMode(_md.NeighborList.storageMode.full)
        self.cpp_force = self.cpp_class(hoomd.context.current.system_definition, self.nlist.cpp_nlist, self.name)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficent options
        self.required_coeffs = ['epsilon', 'sigma', 'lam']

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon']
        sigma = coeff['sigma']
        lam = coeff['lam']

        lj1 = 4.0 * epsilon * math.pow(sigma, 48.0)
        lj2 = 4.0 * epsilon * math.pow(sigma, 24.0)
        rwcasq = math.pow(2.0, 1.0/12.0) * sigma**2
        wca_shift = epsilon * (1. - lam)
        return _azplugins.make_ashbaugh_params(lj1, lj2, lam, rwcasq, wca_shift)

class lj96(hoomd.md.pair.pair):
    R""" LJ 9-6 potential

    Args:
        r_cut (float): Default cutoff radius (in distance units).
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list
        name (str): Name of the force instance.

    :py:class:`lj96` is a Lennard-Jones potential

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V(r)  = & \frac{27}{4} \varepsilon \left(\left(\frac{\sigma}{r}\right)^9 - \alpha \left(\frac{\sigma}{r}\right)^6\right) &, r < r_{\mathrm{cut}} \\
              = & 0 &, r \ge r_{\mathrm{cut}}
        \end{eqnarray*}

    Parameters :math:`\varepsilon`, :math:`\sigma`, and :math:`\alpha`. See :py:class:`hoomd.md.pair.pair`
    for details on how forces are calculated and the available energy shifting and smoothing modes.
    Use :py:meth:`pair_coeff.set <coeff.set>` to set potential coefficients.

    The following coefficients must be set per unique pair of particle types:

    - :math:`\varepsilon` - *epsilon* (in energy units)
    - :math:`\sigma` - *sigma* (in distance units)
    - :math:`\alpha` - *alpha* (unitless) - *optional*: defaults to 1.0
    - :math:`r_{\mathrm{cut}}` - *r_cut* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command

    Example::

        nl = hoomd.md.nlist.cell()
        lj96 = azplugins.pair.lj96(r_cut=3.0, nlist=nl)
        lj96.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        lj96.pair_coeff.set(['A','B'], 'B', epsilon=2.0, sigma=1.0, alpha=0.5)

    """
    def __init__(self, r_cut, nlist, name=None):
        hoomd.util.print_status_line()

        # initialize the base class
        hoomd.md.pair.pair.__init__(self, r_cut, nlist, name)

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_class = _azplugins.PairPotentialLJ96
        else:
            self.cpp_class = _azplugins.PairPotentialLJ96GPU
            self.nlist.cpp_nlist.setStorageMode(_md.NeighborList.storageMode.full)
        self.cpp_force = self.cpp_class(hoomd.context.current.system_definition, self.nlist.cpp_nlist, self.name)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficent options
        self.required_coeffs = ['epsilon', 'sigma', 'alpha']
        self.pair_coeff.set_default_coeff('alpha', 1.0)

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon']
        sigma = coeff['sigma']
        alpha = coeff['alpha']

        lj1 = 27.0 / 4.0 * epsilon * math.pow(sigma, 9.0)
        lj2 = alpha * 27.0 / 4.0 * epsilon * math.pow(sigma, 6.0)

        return _hoomd.make_scalar2(lj1, lj2)

class slj(hoomd.md.pair.pair):
    R""" Core-shifted Lennard-Jones potential

    Args:
        r_cut (float): Default cutoff radius (in distance units).
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list
        name (str): Name of the force instance.

    :py:class:`slj` is Lennard-Jones potential with the core (minimum) shifted by
    an amount :math:`\Delta`. The form of the potential is similar to the standard
    Lennard-Jones potential

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V(r)  = & 4 \varepsilon \left[ \left(\frac{\sigma}{r-\Delta} \right)^12
                  - \alpha \left(\frac{\sigma}{r-\Delta} \right)^6 \right] & r < r_{\rm cut} \\
              = & 0 & r \ge r_{\rm cut}
        \end{eqnarray*}

    Here, :math:`\varepsilon`, :math:`\sigma`, and :math:`\alpha` are the stanard
    Lennard-Jones potential parameters, and :math:`\Delta` is the amount the potential
    is shifted by. The minimum of the potential :math:`r_{\rm min}` is shifted to

    .. math::

        r_{\rm min} = 2^{1/6} \sigma + \Delta

    Setting :math:`\Delta = 0` recovers the standard Lennard-Jones potential.
    See :py:class:`hoomd.md.pair.pair` for details on how forces are calculated
    and the available energy shifting and smoothing modes.
    Use :py:meth:`pair_coeff.set <coeff.set>` to set potential coefficients.

    The following coefficients must be set per unique pair of particle types:

    - :math:`\varepsilon` - *epsilon* (in energy units)
    - :math:`\sigma` - *sigma* (in distance units)
    - :math:`\Delta` - *delta* (in distance units)
    - :math:`\alpha` - *alpha* (unitless) - *optional*: defaults to 1.0
    - :math:`r_{\mathrm{cut}}` - *r_cut* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command
    - :math:`r_{\mathrm{on}}`- *r_on* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command

    Example::

        nl = hoomd.md.nlist.cell()
        slj = azplugins.pair.slj(r_cut=3.0, nlist=nl)
        slj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, delta=1.0)
        slj.pair_coeff.set(['A','B'], 'B', epsilon=2.0, sigma=1.0, alpha=0.5, delta=0.0, r_cut=3.0, r_on=2.0)

    .. note::
        Because of the form of the potential, square-root calls are necessary
        to evaluate the potential and also to perform energy shifting. This will
        incur a corresponding performance hit compared to the standard
        Lennard-Jones potential even when :math:`\Delta=0`.

    """
    def __init__(self, r_cut, nlist, name=None):
        hoomd.util.print_status_line()

        # initialize the base class
        hoomd.md.pair.pair.__init__(self, r_cut, nlist, name)

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_class = _azplugins.PairPotentialShiftedLJ
        else:
            self.cpp_class = _azplugins.PairPotentialShiftedLJGPU
            self.nlist.cpp_nlist.setStorageMode(_md.NeighborList.storageMode.full)
        self.cpp_force = self.cpp_class(hoomd.context.current.system_definition, self.nlist.cpp_nlist, self.name)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficent options
        self.required_coeffs = ['epsilon', 'sigma', 'delta', 'alpha']
        self.pair_coeff.set_default_coeff('alpha', 1.0)

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon']
        sigma = coeff['sigma']
        delta = coeff['delta']
        alpha = coeff['alpha']

        lj1 = 4.0 * epsilon * math.pow(sigma, 12.0)
        lj2 = alpha * 4.0 * epsilon * math.pow(sigma, 6.0)
        return _hoomd.make_scalar3(lj1, lj2, delta)



class spline(hoomd.md.pair.pair):
    R""" Spline potential

    Args:
        r_cut (float): Default cutoff radius (in distance units).
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list
        name (str): Name of the force instance.

    :py:class:`spline` is a spline potential of the form:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V(r) = & a &, r < r_{\rm s}\\
        = & \frac{a(r_{\rm s}^2-r^2)^m (r_{\rm cut}^2 + m r^2 - (m+1) r_{\rm s}^2)}{(r_{\rm cut}^2-r_{\rm s}^2)^{m+1}} &, r_{\rm s} < r < r_{\rm cut} \\
              = & 0 &, r \ge r_{\rm cut}
        \end{eqnarray*}

    Here, :math:`a` is the amplitude :math:`m`, the exponent,  :math:`r_{\rm s}` and :math: `r_{\rm cut}` are the cutoff distances. The potential goes smoothly from a value of `a` at `r_{\rm s}` to zero at
    `r_{\rm cut}`. The slope is determined by the exponent `m`, which needs to be greater or equal to 2.
    The force goes to zero at both the cutoff value `r_{\rm cut}` and `r_{\rm s}`.

    The following coefficients must be set per unique pair of particle types:

    - :math:`a` - *amp* - value of the plateau (in energy units)
    - :math:`m` - *exponent*
    - :math:`r_{\mathrm{cut}}` - *r_cut* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command
    - :math:`r_{\mathrm{s}}`- *r_start* (in distance units) value where the potential reaches its plateau value

    Example::

        nl = hoomd.md.nlist.cell()
        s = azplugins.pair.spline(r_cut=3.0, nlist=nl)
        s.pair_coeff.set('A', 'A', amp=1.0, r_start=1.0, m=2.0)
        s.pair_coeff.set(['A','B'], 'B',  amp=1.0, r_start=1.0, m=2.0, r_cut=3.0)

    """
    def __init__(self, r_cut, nlist, name=None):
        hoomd.util.print_status_line()

        # initialize the base class
        hoomd.md.pair.pair.__init__(self, r_cut, nlist, name)

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_class = _azplugins.PairPotentialSpline
        else:
            self.cpp_class = _azplugins.PairPotentialSplineGPU
            self.nlist.cpp_nlist.setStorageMode(_md.NeighborList.storageMode.full)
        self.cpp_force = self.cpp_class(hoomd.context.current.system_definition, self.nlist.cpp_nlist, self.name)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficent options
        self.required_coeffs = ['amp', 'm', 'r_start']
        self.r_cut=r_cut

    def process_coeff(self, coeff):
        amplitude = coeff['amp']
        exponent = coeff['m']
        r_start = coeff['r_start']
        r_start_sq = r_start*r_start
        if exponent<2.0:
            hoomd.context.msg.error('azplugins.pair.spline: Exponent for spline needs to be >= 2.\n')
            raise ValueError('Exponent for spline needs to be >= 2')
        if r_start>=self.r_cut:
            hoomd.context.msg.error('azplugins.pair.spline: r_start needs to be smaller than r_cut.\n')
            raise ValueError('r_start needs to be smaller than r_cut')
        return _hoomd.make_scalar3(amplitude, exponent,r_start_sq)

class two_patch_morse(hoomd.md.pair.ai_pair):
    R""" Two patches with Morse potential

    Args:
        r_cut (float): Default cutoff radius (in distance units).
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list
        name (str): Name of the force instance.

    :py:class:`two_patch_morse` is a Morse potential which is modulated by an orientation-dependent function.

    .. math::

        V_{\rm M2P}(\vec{r}_{ij}, \hat{n}_i, \hat{n}_j) = V_{\rm M}(|\vec{r}_{ij}|) \Omega(\hat{r}_{ij} \cdot \hat{n}_i) \Omega(\hat{r}_{ij} \cdot \hat{n}_j)

    where :math:`V_{\rm M}` is the potential that depends on distance

    .. math::

        V_{\rm M}(r) = M_d \left( \left[ 1 - \exp\left( -\frac{r-r_{\rm eq}}{M_r}\right) \right]^2 - 1 \right)

    and :math:`\Omega(\gamma)` depends on the orientations

    .. math::
        \Omega(\gamma) = \frac{1}{1+\exp[-\omega (\gamma^2 - \alpha)]}

    The potential can be smoothed to zero force (making it purely attractive) when :math:`r < r_{\rm eq}`
    by making :math:`V_{\rm M}(r < r_{\rm eq}) = -M_d` when the option  *repulsion* is ``False``.

    Here, :math:`vec{r}_{ij}` is the displacement vector between particles :math:`i` and :math:`j`,
    :math:`|\vec{r}_{ij}|` is the magnitude of that displacement, and :math:`\hat{n}` is the normalized
    orientation vector of the particle. The parameters :math:`M_d`, :math:`M_r`, and :math:`r_{\rm eq}`
    control the depth, width, and position of the potential well. The parameters :math:`\alpha` and
    :math:`\omega` control the width and steepness of the orientation dependence.

    Use :py:meth:`pair_coeff.set <coeff.set>` to set potential coefficients.

    The following coefficients must be set per unique pair of particle types:

    - :math:`M_d` - *Md* (in energy units)
    - :math:`M_r` - *Mr* (in distance units)
    - :math:`r_{\rm eq}` - *req* (in distance units)
    - :math:`\omega` - *omega* (unitless)
    - :math:`\alpha` - *alpha* (unitless)
    - *repulsion* (boolean)
    - :math:`r_{\mathrm{cut}}` - *r_cut* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command

    Example::

        nl = hoomd.md.nlist.cell()
        m2p = azplugins.pair.two_patch_morse(r_cut=1.6, nlist=nl)
        m2p.pair_coeff.set('A', 'A', Md=1.8347, Mr=0.0302, req=1.0043, omega=20, alpha=0.50, repulsion=True)

    """
    def __init__(self, r_cut, nlist, name=None):
        hoomd.util.print_status_line()

        # initialize the base class
        hoomd.md.pair.ai_pair.__init__(self, r_cut, nlist, name)

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_class = _azplugins.AnisoPairPotentialTwoPatchMorse
        else:
            self.cpp_class = _azplugins.AnisoPairPotentialTwoPatchMorseGPU
            self.nlist.cpp_nlist.setStorageMode(_md.NeighborList.storageMode.full)
        self.cpp_force = self.cpp_class(hoomd.context.current.system_definition, self.nlist.cpp_nlist, self.name)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficients
        self.required_coeffs = ['Md', 'Mr', 'req', 'omega', 'alpha','repulsion']
        self.pair_coeff.set_default_coeff('repulsion', True)

    def process_coeff(self, coeff):
        Md        = coeff['Md']
        Mr        = coeff['Mr']
        req       = coeff['req']
        omega     = coeff['omega']
        alpha     = coeff['alpha']
        repulsion = coeff['repulsion']

        return _azplugins.make_two_patch_morse_params(Md, 1.0/Mr, req, omega, alpha, repulsion)
