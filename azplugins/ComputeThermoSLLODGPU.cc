// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file ComputeThermoSLLODGPU.cc
    \brief Contains code for the ComputeThermoSLLODGPU class
*/


#include "ComputeThermoSLLODGPU.h"
#include "ComputeThermoSLLODGPU.cuh"
#include "hoomd/GPUPartition.cuh"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#include "hoomd/HOOMDMPI.h"
#endif

#include <iostream>

namespace azplugins
{
/*! \param sysdef System for which to compute thermodynamic properties
    \param group Subset of the system over which properties are calculated
    \param suffix Suffix to append to all logged quantity names
*/

ComputeThermoSLLODGPU::ComputeThermoSLLODGPU(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<ParticleGroup> group,
                                   Scalar shear_rate,
                                   const std::string& suffix)
    : ComputeThermoSLLOD(sysdef, group, shear_rate, suffix), m_scratch(m_exec_conf), m_scratch_pressure_tensor(m_exec_conf),
        m_scratch_rot(m_exec_conf)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a ComputeThermoSLLODGPU with no GPU in the execution configuration" << std::endl;
        throw std::runtime_error("Error initializing ComputeThermoSLLODGPU");
        }

    m_block_size = 512;

    cudaEventCreate(&m_event, cudaEventDisableTiming);
    }

//! Destructor
ComputeThermoSLLODGPU::~ComputeThermoSLLODGPU()
    {
    cudaEventDestroy(m_event);
    }

/*! Computes all thermodynamic properties of the system in one fell swoop, on the GPU.
 */
void ComputeThermoSLLODGPU::computeProperties()
    {
    // just drop out if the group is an empty group
    if (m_group->getNumMembersGlobal() == 0)
        return;

    unsigned int group_size = m_group->getNumMembers();

    removeFlowField();

    if (m_prof) m_prof->push(m_exec_conf,"Thermo");

    assert(m_pdata);
    assert(m_ndof != 0);

    // number of blocks in reduction (round up for every GPU)
    unsigned int num_blocks = m_group->getNumMembers() / m_block_size + m_exec_conf->getNumActiveGPUs();

    // resize work space
    unsigned int old_size = m_scratch.size();

    m_scratch.resize(num_blocks);
    m_scratch_pressure_tensor.resize(num_blocks*6);
    m_scratch_rot.resize(num_blocks);

    if (m_scratch.size() != old_size)
        {
        if (m_exec_conf->allConcurrentManagedAccess())
            {
            auto& gpu_map  = m_exec_conf->getGPUIds();

            // map scratch array into memory of all GPUs
            for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
                {
                cudaMemAdvise(m_scratch.get(), sizeof(Scalar4)*m_scratch.getNumElements(), cudaMemAdviseSetAccessedBy, gpu_map[idev]);
                cudaMemAdvise(m_scratch_pressure_tensor.get(), sizeof(Scalar)*m_scratch_pressure_tensor.getNumElements(), cudaMemAdviseSetAccessedBy, gpu_map[idev]);
                cudaMemAdvise(m_scratch_rot.get(), sizeof(Scalar)*m_scratch_rot.getNumElements(), cudaMemAdviseSetAccessedBy, gpu_map[idev]);
                }
            CHECK_CUDA_ERROR();
            }

        // reset to zero, to be on the safe side
        ArrayHandle<Scalar4> d_scratch(m_scratch, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_scratch_pressure_tensor(m_scratch_pressure_tensor, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_scratch_rot(m_scratch_rot, access_location::device, access_mode::overwrite);

        cudaMemset(d_scratch.data, 0, sizeof(Scalar4)*m_scratch.size());
        cudaMemset(d_scratch_pressure_tensor.data, 0, sizeof(Scalar)*m_scratch_pressure_tensor.size());
        cudaMemset(d_scratch_rot.data, 0, sizeof(Scalar)*m_scratch_rot.size());
        }

    // access the particle data
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getGlobalBox();

    PDataFlags flags = m_pdata->getFlags();

    { // scope these array handles so they are released before the additional terms are added
    // access the net force, pe, and virial
    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();
    const GlobalArray< Scalar >& net_virial = m_pdata->getNetVirial();
    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_net_virial(net_virial, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_scratch(m_scratch, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_scratch_pressure_tensor(m_scratch_pressure_tensor, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_scratch_rot(m_scratch_rot, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_properties(m_properties, access_location::device, access_mode::overwrite);

    // access the group
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

    m_exec_conf->beginMultiGPU();

    // build up args list
    compute_thermo_args args;
    args.n_blocks = num_blocks;
    args.d_net_force = d_net_force.data;
    args.d_net_virial = d_net_virial.data;
    args.d_orientation = d_orientation.data;
    args.d_angmom = d_angmom.data;
    args.d_inertia = d_inertia.data;
    args.virial_pitch = net_virial.getPitch();
    args.ndof = m_ndof;
    args.D = m_sysdef->getNDimensions();
    args.d_scratch = d_scratch.data;
    args.d_scratch_pressure_tensor = d_scratch_pressure_tensor.data;
    args.d_scratch_rot = d_scratch_rot.data;
    args.block_size = m_block_size;
    args.external_virial_xx = m_pdata->getExternalVirial(0);
    args.external_virial_xy = m_pdata->getExternalVirial(1);
    args.external_virial_xz = m_pdata->getExternalVirial(2);
    args.external_virial_yy = m_pdata->getExternalVirial(3);
    args.external_virial_yz = m_pdata->getExternalVirial(4);
    args.external_virial_zz = m_pdata->getExternalVirial(5);
    args.external_energy = m_pdata->getExternalEnergy();

    // perform the computation on the GPU(s)
    gpu::compute_thermo_partial( d_properties.data,
                        d_vel.data,
                        d_body.data,
                        d_tag.data,
                        d_index_array.data,
                        group_size,
                        box,
                        args,
                        flags[pdata_flag::pressure_tensor],
                        flags[pdata_flag::rotational_kinetic_energy],
                        m_group->getGPUPartition());

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    // converge GPUs
    m_exec_conf->endMultiGPU();

    // perform the computation on GPU 0
    gpu::compute_thermo_final( d_properties.data,
                        d_vel.data,
                        d_body.data,
                        d_tag.data,
                        d_index_array.data,
                        group_size,
                        box,
                        args,
                        flags[pdata_flag::pressure_tensor],
                        flags[pdata_flag::rotational_kinetic_energy]);

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    }

    #ifdef ENABLE_MPI
    // in MPI, reduce extensive quantities only when they're needed
    m_properties_reduced = !m_pdata->getDomainDecomposition();
    #endif // ENABLE_MPI

    addFlowField();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

#ifdef ENABLE_MPI
void ComputeThermoSLLODGPU::reduceProperties()
    {
    if (m_properties_reduced) return;

    ArrayHandle<Scalar> h_properties(m_properties, access_location::host, access_mode::readwrite);

    // reduce properties
    MPI_Allreduce(MPI_IN_PLACE, h_properties.data, thermo_index::num_quantities, MPI_HOOMD_SCALAR,
            MPI_SUM, m_exec_conf->getMPICommunicator());

    m_properties_reduced = true;
    }
#endif

void ComputeThermoSLLODGPU::removeFlowField()
{
  unsigned int group_size = m_group->getNumMembers();

  // profile this step
  if (m_prof)
      m_prof->push("remove flowfield");

  ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
  ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
  ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

  m_exec_conf->beginMultiGPU();

  // perform the removal of the flow field on the GPU

  gpu::remove_flow_field(d_vel.data,
                   d_pos.data,
                   d_index_array.data,
                   group_size,
                   m_block_size,
                   m_shear_rate,
                   m_group->getGPUPartition());


  if(m_exec_conf->isCUDAErrorCheckingEnabled())
      CHECK_CUDA_ERROR();

  m_exec_conf->endMultiGPU();


  // done profiling
  if (m_prof)
      m_prof->pop();
}

void ComputeThermoSLLODGPU::addFlowField()
{
  unsigned int group_size = m_group->getNumMembers();

  // profile this step
  if (m_prof)
      m_prof->push("SLLOD NVT remove flowfield");

  ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
  ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
  ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

  m_exec_conf->beginMultiGPU();

  // perform the removal of the flow field on the GPU
  gpu::add_flow_field(d_vel.data,
                   d_pos.data,
                   d_index_array.data,
                   group_size,
                   m_block_size,
                   m_shear_rate,
                   m_group->getGPUPartition());


  if(m_exec_conf->isCUDAErrorCheckingEnabled())
      CHECK_CUDA_ERROR();


  m_exec_conf->endMultiGPU();


  // done profiling
  if (m_prof)
      m_prof->pop();
}

namespace detail
{

void export_ComputeThermoSLLODGPU(pybind11::module& m)
    {
    pybind11::class_<ComputeThermoSLLODGPU, std::shared_ptr<ComputeThermoSLLODGPU> >(m,"ComputeThermoSLLODGPU",pybind11::base<ComputeThermoSLLOD>())
    .def(pybind11::init< std::shared_ptr<SystemDefinition>,
         std::shared_ptr<ParticleGroup>,
         Scalar,
         const std::string& >())
        ;
    }

} // end namespace detail
} // end namespace azplugins
