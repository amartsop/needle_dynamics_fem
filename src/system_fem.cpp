#include "system_fem.h"


SystemFem::SystemFem(RigidBody *handle, FlexibleBeamFem *needle, 
    InputTrajectory *input_traj)
{
    // Rigid body (handle)
    m_handle_ptr = handle;

    // Flexible body (needle)
    m_needle_ptr = needle;

    // Input coordinates 
    m_input_trajectory = input_traj;

    // Number of elements
    m_elements = needle->get_elements_number();

    // Number of dofs
    m_dofs = 5 * (m_elements + 1) + 6; 

    // All coefficients
    m_l = arma::regspace<arma::imat>(1, 1, m_dofs);

    // Indices of uknown coordinates
    m_la = arma::regspace<arma::imat>(6 + (10 / 2) + 1, 1, m_dofs);

    // Size of qa vector 
    m_qa_size = m_la.n_rows;

}

arma::dvec SystemFem::calculate(arma::dvec state_vector, double t)
{
    // Uknown coordinates
    arma::dvec qa = state_vector.rows(0, m_qa_size - 1);
    arma::dvec qa_dot = state_vector.rows(m_qa_size, 2 * m_qa_size - 1);

    // Known coordinates
    m_input_trajectory->update(t);
    arma::dvec qg = m_input_trajectory->get_displacement_qg();
    arma::dvec qg_dot = m_input_trajectory->get_velocity_qg_dot();
    arma::dvec qg_ddot = m_input_trajectory->get_acceleration_qg_ddot();

    // Full vector 
    arma::dvec q = arma::join_vert(qg, qa);
    arma::dvec q_dot = arma::join_vert(qg_dot, qa_dot);

    // Update mass, stiffness matrix and coriolis vector
    update(t, q, q_dot);

    //************ Boundary conditions ************//

    // First and last indices of known coordinates 
    uint lg_first = m_lg.front() - 1;
    uint lg_last = m_lg.back() - 1;
    
    // First and last indices of ukknown coordinates 
    uint la_first = m_la.front() - 1;
    uint la_last = m_la.back() - 1;

    // Known and uknown mass matrices 
    m_mgg = m_mass(arma::span(lg_first, lg_last), arma::span(lg_first, lg_last));
    m_mga = m_mass(arma::span(lg_first, lg_last), arma::span(la_first, la_last));
    m_mag = m_mass(arma::span(la_first, la_last), arma::span(lg_first, lg_last));
    m_maa = m_mass(arma::span(la_first, la_last), arma::span(la_first, la_last));

    // Known and uknown stiffness matrices 
    m_kgg = m_stiffness(arma::span(lg_first, lg_last), arma::span(lg_first, lg_last));
    m_kga = m_stiffness(arma::span(lg_first, lg_last), arma::span(la_first, la_last));
    m_kag = m_stiffness(arma::span(la_first, la_last), arma::span(lg_first, lg_last));
    m_kaa = m_stiffness(arma::span(la_first, la_last), arma::span(la_first, la_last));

    // Known and uknown coriolis vector 
    m_fvg = m_fv.rows(lg_first, lg_last);
    m_fva = m_fv.rows(la_first, la_last);

    // External forces corresponding to uknown coordinates 
    arma::dvec qfj = m_needle_ptr->get_external_force_vector();
    m_qa_force = qfj.rows(la_first, la_last);

    //************ Model ************//
    arma::dvec x1 = qa;
    arma::dvec x2 = qa_dot;

    // Reaction forces
    arma::dvec qa_ddot = arma::solve(m_maa, m_qa_force + m_fva - m_kaa * x1
        - m_mag * qg_ddot - m_kag * qg);

    m_qg_force = m_mgg * qg_ddot + m_mga * qa_ddot + m_kgg * qg + m_kga * qa 
        - m_fvg;


    // System model
    arma::dvec x1_dot = x2;
    arma::dvec x2_dot = qa_ddot;


    return arma::join_vert(x1_dot, x2_dot);
}


void SystemFem::update(double t, arma::dvec q, arma::dvec q_dot)
{
    // Update mass, stiffness, coriolis and external forces 
    m_handle_ptr->update(t, q, q_dot);
    m_needle_ptr->update(t, q, q_dot);

    // Total mass matrix
    m_mass = m_handle_ptr->get_mass_matrix() + m_needle_ptr->get_mass_matrix();

    // Total stiffness matrix 
    m_stiffness = m_needle_ptr->get_stiffness_matrix();

    // Total coriolis force matrix 
    m_fv = m_handle_ptr->get_coriolis_vector() + m_needle_ptr->get_coriolis_vector();
}


