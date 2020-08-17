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

    // Calculate system's Jacobian
    arma::dmat dfdx(double t, arma::dvec x);

    // Get reaction forces 
    arma::dvec get_reaction_forces(void) { return m_fc_f; }

    // Get reaction forces 
    arma::dvec get_reaction_moment(void) { return m_mc_f; }

private:

    // Number of elastic dofs
    uint m_elastic_dofs;

    // Raction force 
    arma::dvec m_fc_f;

    // Raction Moment
    arma::dvec m_mc_f;

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

    // Elastic damping matrix calculation
    void elastic_damping_matrix_calculation(void);

    // Total damping matrix calculation
    void total_damping_matrix_calculation(void);
    
    // Total damping matrix
    arma::dmat m_cf;

    // Transformation matrix 
    arma::dmat m_p_mat;

    // First two frequencies 
    arma::dvec m_freq = arma::zeros<arma::dvec>(2, 1);

    // Systen eigenvalues 
    arma::dvec m_eigval;

    // Elastic mass, stiffness and damping matrices
    arma::dmat m_mf33, m_kf33, m_cf33;
    
    // Transformed elastic mass, stiffness and damping matrices
    arma::dmat m_mf33_tilde, m_kf33_tilde, m_cf33_tilde;

    // Damping coeefficients 
    double m_mu, m_kappa;

    // Damping ratio
    const double m_zeta1 = 0.1;
    const double m_zeta2 = 0.1;

private:
    // Numerical Jacobian tolerance
    double m_tol = 1.0e-8;

};

