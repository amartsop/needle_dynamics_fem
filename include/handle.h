#pragma once

#include <iostream>
#include <armadillo>

#include "dynamics_math.h"
#include "needle_properties.hpp"


class Handle : public NeedleProperties 
{

public:
    Handle();

    // Update rigid body matrices and coriolis vector
    void update(double t, arma::dvec q, arma::dvec q_dot);

    // Mass matrix getter
    arma::dmat get_mass_matrix(void){ return m_mass; }
    
    // Coriolis vector getter
    arma::dvec get_coriolis_vector(void){ return m_fvr; }

    // External force vector getter
    arma::dvec get_external_forcel(void){ return m_qr; }

    // Get rotation matrix 
    arma::dmat get_rotation_matrix(void) { return m_rot_f_F; }
    
    // Get g matrix 
    arma::dmat get_g_matrix(void) { return m_g_mat; }

    // Get g_dot matrix 
    arma::dmat get_g_dot_matrix(void) { return m_g_dot_mat; }

    // Get total handle dofs 
    uint get_handle_dofs(void) { return m_dofs; }

    // Get weight force 
    arma::dvec get_weight_force(void) { return m_weight; }

    // Get frames distance 
    arma::dvec get_rca_f_f(void) { return m_rca_f_f; }

    // Get handle inertia tensor
    arma::dmat get_inertial_tensor(void) { return m_i_c_f; }

private:
    // Handle length 
    double m_length;

    // Handle radius 
    double m_radius;

    // Handle mass 
    double m_mr;

    // Handle weight force inertial 
    arma::dvec m_weight;

    // Handle inertia wrt to centre of mass 
    arma::dmat m_i_c_f;

    // Distance between rigid body's com and reference frame a 
    arma::dvec m_rca_f_f;

private:
    // Number of rigid dofs
    const uint m_rigid_dofs = 6;

    // Number of dofs
    uint m_dofs;

private:
    // State 
    arma::dvec m_roc_F_F, m_theta;

    // State dot 
    arma::dvec m_roc_dot_F_F, m_theta_dot;

    // Rotational velocity, G, Gdot and rotation matrices
    arma::dmat m_omega, m_g_mat, m_g_dot_mat, m_rot_f_F;

    // Current time 
    double m_time;

private:
    // Rigid body mass matrix
    arma::dmat m_mass;

    // Coriolis-Centrifugal vector
    arma::dvec m_fvr;

    // External forces 
    arma::dvec m_qr;

private:
    // State update 
    void state_update(arma::dvec q, arma::dvec q_dot);

    // Mass matrix calculation
    void mass_matrix_calculation(void); 

    // Coriolis-Centrifugal calculation
    void coriolis_vector_calculation(void);

};