#include "handle.h"
#include "euler_rotations.h"

Handle::Handle()
{
    // Handle length
    m_length = m_handle_length;

    // Handle radius
    m_radius = m_handle_radius;

    // Handle mass 
    m_mr = m_handle_mass;

    // Handle weight force inertial 
    m_weight = {0.0, 0.0, - m_grav * m_mr};
    
    // Handle inertia wrt to centre of mass 
    m_i_c_f = {{0.5, 0.0, 0.0}, 
        {0.0, 0.25 + (1.0 / 12.0) * pow((m_length / m_radius), 2.0), 0.0},
        {0.0, 0.0, 0.25 + (1.0 / 12.0) * pow((m_length / m_radius), 2.0)}};
    m_i_c_f *= (m_mr * pow(m_radius, 2.0));

    // Distance between rigid body's com and reference frame a 
    m_rca_f_f = {m_needle_lx, 0.0, - m_needle_lz};

    // Number of dofs 
    m_dofs = m_rigid_dofs;
}

void Handle::state_update(arma::dvec q, arma::dvec q_dot)
{
    // State
    m_roc_F_F = {q(0), q(1), q(2)};
    m_theta = {q(3), q(4), q(5)};

    // State dot
    m_roc_dot_F_F = {q_dot(0), q_dot(1), q_dot(2)};
    m_theta_dot = {q_dot(3), q_dot(4), q_dot(5)};
}

void Handle::update(double t, arma::dvec q, arma::dvec q_dot)
{
    // State Update 
    state_update(q, q_dot);

    // G and Gdot matrices
    m_g_mat = EulerRotations::G(m_theta);
    m_g_dot_mat = EulerRotations::G_dot(m_theta, m_theta_dot);

    // Rotation matrix 
    m_rot_f_F = EulerRotations::rotation(m_theta);

    // Angular velocity (rad / sec)
    m_omega =  m_g_mat * m_theta_dot;

    // Time 
    m_time = t;

    // Update mass matrix
    mass_matrix_calculation();
    
    // Update coriolis-centrifugal vector
    coriolis_vector_calculation();
}

void Handle::mass_matrix_calculation(void)
{
    // First row
    arma::dmat mr11 = m_mr * arma::eye(3, 3);
    arma::dmat mr12 = arma::zeros(3, 3);
    arma::dmat mr1 = arma::join_horiz(mr11, mr12);

    // Second row 
    arma::dmat mr21 = mr12.t();
    arma::dmat mr22 = m_g_mat.t() * m_i_c_f * m_g_mat;
    arma::dmat mr2 = arma::join_horiz(mr21, mr22);

    // Mass matrix
    m_mass = arma::join_vert(mr1, mr2);
}


void Handle::coriolis_vector_calculation(void)
{
    // First vector
    arma::dvec fvr1 = arma::zeros(3, 1);

    // Second vector 
    arma::dvec fvr2 = - m_g_mat.t() * ( dm::s(m_omega) *
        (m_i_c_f * m_omega) + m_i_c_f * m_g_dot_mat * m_theta_dot );

    // Coriolis-Centrifugal vector
    m_fvr = arma::join_vert(fvr1, fvr2);
}

