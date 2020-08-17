#pragma once 

#include <iostream>

#include "handle.h"
#include "flexible_beam_fem.h"
#include "input_trajectory.h"


class SystemFem
{
public:
    SystemFem(Handle *handle, FlexibleBeamFem *needle,
        InputTrajectory *input_traj);

    // Calculate system model function 
    arma::dvec f(double t, arma::dvec state_vector);

    // // Get reaction forces 
    // arma::dvec get_reaction_forces(void) { return m_qg_force; }

    // // Get uknown vector size 
    // uint get_qa_size(void) { return m_qa_size; }

    // // Get model size 
    // uint get_model_size(void) {return 2 * m_qa_size; }

private:

    // Number of elastic dofs
    uint m_elastic_dofs;

    // // Number of elements
    // uint m_elements;

    // // Number of dofs
    // uint m_dofs;

    // // Indices of all coordinates 
    // arma::ivec m_l;

    // // Indices of uknown coordinates
    // arma::ivec m_la;

    // // Indices of known coordinates 
    // const arma::ivec m_lg = arma::regspace<arma::ivec>(1, 1, 6 + (10 / 2));

    // // Size of qa vector 
    // uint m_qa_size;

private:
    // Rigid body (handle)
    Handle* m_handle_ptr;

    // Flexible body (needle)
    FlexibleBeamFem* m_needle_ptr; 

    // Input coordinates 
    InputTrajectory* m_input_traj_ptr;

private:
    // Calculate coordinates transformation matrix 
    void coordinate_transformation_matrix(void);

    // Transformation matrix 
    arma::dmat m_p_mat;

    // First two frequencies 
    arma::dvec m_freq = arma::zeros<arma::dvec>(2, 1);

    // Elastic mass and stiffness matrices
    arma::dmat m_mf33, m_kf33;

// private:
//     // System mass and stiffness matrix
//     arma::dmat m_mass, m_stiffness;
    
//     // System coriolis-centrifugal and external forces vector
//     arma::dvec m_fv;


// private:
//     // Mass matrix of known and uknown coefficients
//     arma::dmat m_mgg, m_mga, m_mag, m_maa;

//     // Stiffness matrix of known and uknown coefficients
//     arma::dmat m_kgg, m_kga, m_kag, m_kaa;

//     // Coriolis vector of known and uknown coefficients 
//     arma::dvec m_fva, m_fvg;

//     // External forces vector of known and uknown coefficients 
//     arma::dvec m_qa_force, m_qg_force;

// private:
//     // Update matrices and vectors 
//     void update(double t, arma::dvec q, arma::dvec q_dot);
};

