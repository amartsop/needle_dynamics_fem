#include "rigid_body.h"
#include "euler_rotations.h"

RigidBody::RigidBody(/* args */)
{
    // Handle length
    m_length = m_handle_length;

    // Handle radius
    m_radius = m_handle_radius;

    // Distance vector of reference frame origin and centre of mass (wrt reference frame)
    m_r_ac_f_f = {-m_needle_lx, 0.0, m_needle_lz};

    // Handle mass 
    m_mr = m_handle_mass;

    // Handle inertia wrt to centre of mass 
    arma::dmat inertia_centre_of_mass = {{0.5, 0.0, 0.0}, 
        {0.0, 0.25 + (1.0 / 12.0) * pow((m_length / m_radius), 2.0), 0.0},
        {0.0, 0.0, 0.25 + (1.0 / 12.0) * pow((m_length / m_radius), 2.0)}};
    inertia_centre_of_mass *= (m_mr * pow(m_radius, 2.0));

    // Handle inertia wrt to reference frame
    m_ir = inertia_centre_of_mass + m_mr * dm::s(m_r_ac_f_f).t() * dm::s(m_r_ac_f_f);
}


void RigidBody::update(double t, arma::dvec q, arma::dvec q_dot)
{
    // Update mass matrix
    mass_matrix_calculation(q, q_dot);
    
    // Update coriolis-centrifugal vector
    coriolis_vector_calculation(q, q_dot);
}

void RigidBody::mass_matrix_calculation(arma::dvec q, arma::dvec q_dot)
{
    // State 
    arma::dvec theta = state::theta(q, q_dot);

    // Rotation and G matrix
    arma::dmat rot_f_F = EulerRotations::rotation(theta);
    arma::dmat g_mat = EulerRotations::G(theta);

    // First row
    arma::dmat mr11 = m_mr * arma::eye(3, 3);
    arma::dmat mr12 = - rot_f_F * m_mr * dm::s(m_r_ac_f_f) * g_mat;
    arma::dmat mr13 = arma::zeros(3, state::nf(q));
    arma::dmat mr1 = arma::join_horiz(mr11, mr12, mr13);

    // Second row 
    arma::dmat mr21 = mr12.t();
    arma::dmat mr22 = g_mat.t() * m_ir * g_mat;
    arma::dmat mr23 = arma::zeros(3, state::nf(q));
    arma::dmat mr2 = arma::join_horiz(mr21, mr22, mr23);

    // Third row 
    arma::dmat mr31 = arma::zeros(state::nf(q), 3);
    arma::dmat mr32 = arma::zeros(state::nf(q), 3);
    arma::dmat mr33 = arma::zeros(state::nf(q), state::nf(q));
    arma::dmat mr3 = arma::join_horiz(mr31, mr32, mr33);

    // Mass matrix
    m_mass = arma::join_vert(mr1, mr2, mr3);
}


void RigidBody::coriolis_vector_calculation(arma::dvec q, arma::dvec q_dot)
{
    // State 
    arma::dvec theta = state::theta(q, q_dot);

    // State dot
    arma::dvec theta_dot = state::theta_dot(q, q_dot);

    // Rotation, G and Gdot matrix
    arma::dmat rot_f_F = EulerRotations::rotation(theta);
    arma::dmat g_mat = EulerRotations::G(theta);
    arma::dmat g_dot_mat = EulerRotations::G_dot(theta, theta_dot);

    // Angular velocity (rad / sec)
    arma::dvec omega =  g_mat * theta_dot;
    
    // First vector
    arma::dvec fvr1 = rot_f_F * m_mr * ( dm::s(m_r_ac_f_f) * g_dot_mat * theta_dot - 
        dm::s(omega) * dm::s(omega) * m_r_ac_f_f );

    // Second vector 
    arma::dvec fvr2 = - g_mat.t() * ( dm::s(omega) * (m_ir * omega) + m_ir * 
        g_dot_mat * theta_dot );

    // Third vector 
    arma::dvec fvr3 = arma::zeros(state::nf(q), 1);

    // Coriolis-Centrifugal vector
    m_fvr = arma::join_vert(fvr1, fvr2, fvr3);
}