#include "system_fem.h"


SystemFem::SystemFem(Handle *handle, FlexibleBeamFem *needle, 
    InputTrajectory *input_traj)
{ // Rigid body (handle)
    m_handle_ptr = handle;

    // Flexible body (needle)
    m_needle_ptr = needle;

    // Input coordinates 
    m_input_traj_ptr = input_traj;

    // Number of elastic dofs 
    m_elastic_dofs = needle->get_elastic_dofs();

    // Elastic mass and stiffness matrices
    m_mf33 = m_needle_ptr->get_mf33_matrix();
    m_kf33 = m_needle_ptr->get_kf33_matrix();

    // Calculate coordinates transformation matrix 
    coordinate_transformation_matrix();

    // Transformed elastic mass and stiffness matrices
    m_mf33_tilde = m_p_mat.t() * m_mf33 * m_p_mat;
    m_kf33_tilde = m_p_mat.t() * m_kf33 * m_p_mat;

    // First two frequencies
    m_freq(0) = sqrt(arma::as_scalar(m_eigval(0)));
    m_freq(1) = sqrt(arma::as_scalar(m_eigval(2)));

    // Calculate damping elastic matrix
    elastic_damping_matrix_calculation();

    // Calculate total damping matrix
    total_damping_matrix_calculation();

}

arma::dvec SystemFem::f(double t, arma::dvec state_vector)
{
    // Update rigid body trajectory
    m_input_traj_ptr->update(t);

    /***************** Rigid body state ***********************/
    // Displacement
    arma::dvec roc_F_F = m_input_traj_ptr->get_linear_displacement();
    arma::dvec theta_r = m_input_traj_ptr->get_rotational_displacement();
    arma::dvec qr = arma::join_vert(roc_F_F, theta_r);

    // Velocity
    arma::dvec roc_dot_F_F = m_input_traj_ptr->get_linear_velocity();
    arma::dvec theta_dot_r = m_input_traj_ptr->get_rotational_velocity();
    arma::dvec qr_dot = arma::join_vert(roc_dot_F_F, theta_dot_r);

    // Acceleration
    arma::dvec roc_ddot_F_F = m_input_traj_ptr->get_linear_acceleration();
    arma::dvec theta_ddot_r = m_input_traj_ptr->get_rotational_acceleration();
    arma::dvec qr_ddot = arma::join_vert(roc_ddot_F_F, theta_ddot_r);

    // Update rigid body matrices 
    m_handle_ptr->update(t, qr, qr_dot);

    // Rotation matrix
    arma::dmat rot_f_F = m_handle_ptr->get_rotation_matrix();

    // G and g_dot matrix matrix 
    arma::dmat g_mat = m_handle_ptr->get_g_matrix();
    arma::dmat g_dot_mat = m_handle_ptr->get_g_dot_matrix();

    // Distance between frames 
    arma::dvec rca_f_f = m_handle_ptr->get_rca_f_f();

    // Omega 
    arma::dvec omega = g_mat * theta_dot_r;

    /***************** Flexible body state ***********************/
    // Displacement
    arma::dvec roa_F_F = roc_F_F + rot_f_F * rca_f_f;
    arma::dvec theta_f = theta_r;
    arma::dvec qf = state_vector.rows(0, m_elastic_dofs - 1);
    arma::dvec q = arma::join_vert(roa_F_F, theta_f, qf);

    // Velocity
    arma::dvec roa_dot_F_F = roc_dot_F_F - rot_f_F * dm::s(rca_f_f) * omega;
    arma::dvec theta_dot_f = theta_dot_r;
    arma::dvec qf_dot = state_vector.rows(m_elastic_dofs, 2 * m_elastic_dofs - 1);
    arma::dvec q_dot = arma::join_vert(roa_dot_F_F, theta_dot_f, qf_dot);

    // Update flexible body matrices 
    m_needle_ptr->update(t, q, q_dot);

    // Get mass matrices
    arma::dmat mf31 = m_needle_ptr->get_mf31_matrix();
    arma::dmat mf32 = m_needle_ptr->get_mf32_matrix();

    // Get fvf3 vector
    arma::dvec fvf3 = m_needle_ptr->get_fvf3_vector();

    // Get qf3 vector
    arma::dvec qf3 = m_needle_ptr->get_qf3_vector();

    // Acceleration
    arma::dvec roa_ddot_F_F = roc_ddot_F_F + rot_f_F * (dm::s(omega) + 
        dm::s(omega) * dm::s(omega)) * rca_f_f;
    arma::dvec theta_ddot_f = theta_ddot_r;

    // Calculate tau3
    arma::dvec tau3 = fvf3 + qf3 - mf31 * roa_ddot_F_F - mf32 * theta_ddot_f;

    /************************ Coordinate transformation *******************/
    // State transformation 
    arma::dvec qf_tilde = m_p_mat.t() * m_mf33 * qf;
    arma::dvec qf_tilde_dot = m_p_mat.t() * m_mf33 * qf_dot;

    // Force vector transformation 
    arma::dvec tau3_tilde = m_p_mat.t() * tau3;

    // Calculate elastic acceleration (transformed)
    arma::dvec qf_tilde_ddot = arma::solve(m_mf33_tilde, tau3_tilde -
        m_cf33_tilde * qf_tilde_dot - m_kf33_tilde * qf_tilde);

    // Calculate elastic acceleration
    arma::dvec qf_ddot = m_p_mat * qf_tilde_ddot;

    // Total Acceleration
    arma::dvec q_ddot = arma::join_vert(roa_ddot_F_F, theta_ddot_f, qf_ddot);

    /********************** Reaction forces *************************/
    arma::dmat mf = m_needle_ptr->get_mass_matrix();
    arma::dmat kf = m_needle_ptr->get_stiffness_matrix();
    arma::dvec fvf = m_needle_ptr->get_coriolis_vector();
    arma::dvec qforce = m_needle_ptr->get_external_force_vector();

    // Total reaction force
    arma::dvec reaction_force = mf * q_ddot + m_cf * q_dot + kf * q - fvf - qforce;

    // Reaction force at point a 
    arma::dvec qc1 = {reaction_force(0), reaction_force(1), reaction_force(2)};
    arma::dvec qc2 = {reaction_force(3), reaction_force(4), reaction_force(5)};
    arma::dvec fa_F = - qc1; arma::dvec ma_f = - arma::solve(g_mat.t(), qc2);

    // Reaction force at point c
    arma::dvec fc_F = m_handle_ptr->get_handle_mass() * roc_ddot_F_F -
        m_handle_ptr->get_weight_force() - fa_F;

    m_fc_f = rot_f_F.t() * fc_F;

    // Handle inertial tensor 
    arma::dmat ic_f = m_handle_ptr->get_inertial_tensor(); 

    // Reaction moment at point c
    m_mc_f = ic_f * (g_dot_mat * theta_r + g_mat * theta_ddot_r) - ma_f -
        dm::s(m_handle_ptr->get_rca_f_f()) * rot_f_F.t() * fa_F -
        dm::s(omega) * ic_f * omega;


    return arma::join_vert(qf_dot, qf_ddot);
}


// Numerical estimation of system's jacobian
arma::dmat SystemFem::dfdx(double t, arma::dvec x)
{
    arma::dvec fx = f(t, x); arma::dvec x_pert = x;
    arma::dmat jacobian = arma::zeros(fx.n_rows, x.n_rows);

    for (uint i = 0; i < x.n_rows; i ++)
    {
        x_pert(i) = x_pert(i) + m_tol;
        jacobian.col(i) = (f(t, x_pert) - fx) / m_tol;
        x_pert(i) = x(i);
    }
    return jacobian;
}


// Elastic damping matrix
void SystemFem::elastic_damping_matrix_calculation(void)
{
    double omega1 = arma::as_scalar(m_freq(0));
    double omega2 = arma::as_scalar(m_freq(1));

    // Rayleigh Damping
    m_kappa = 2.0 * (m_zeta2 * omega2 - m_zeta1 * omega1) /
        (pow(omega2, 2.0) - pow(omega1, 2.0));

    m_mu = 2.0 * m_zeta1 * omega1 - pow(omega1, 2.0) * m_kappa;

    // Transformed damping matrix 
    m_cf33_tilde = m_mu * m_mf33_tilde + m_kappa * m_kf33_tilde;

    // Damping matrix
    m_cf33 = m_mf33 * m_p_mat * m_cf33_tilde * m_p_mat.t() * m_mf33;
}


// Total damping matrix
void SystemFem::total_damping_matrix_calculation(void)
{
    // First row 
    arma::dmat cf11 = arma::zeros(3, 3); arma::dmat cf12 = arma::zeros(3, 3);
    arma::dmat cf13 = arma::zeros(3, m_elastic_dofs);
    arma::dmat cf1 = arma::join_horiz(cf11, cf12, cf13);

    // Second row 
    arma::dmat cf21 = arma::zeros(3, 3); arma::dmat cf22 = arma::zeros(3, 3);
    arma::dmat cf23 = arma::zeros(3, m_elastic_dofs);
    arma::dmat cf2 = arma::join_horiz(cf21, cf22, cf23);

    // Third row 
    arma::dmat cf31 = arma::zeros(m_elastic_dofs, 3);
    arma::dmat cf32 = arma::zeros(m_elastic_dofs, 3);
    arma::dmat cf3 = arma::join_horiz(cf31, cf32, m_cf33);

    // Total damping matrix 
    m_cf = arma::join_vert(cf1, cf2, cf3);
}


// Calculate coordinates transformation matrix 
void SystemFem::coordinate_transformation_matrix(void)
{
    // Unsorted eigevalues and eigenvectors
    arma::cx_vec eigval_gu; arma::cx_mat eigvec_gu;
    arma::eig_pair(eigval_gu, eigvec_gu, m_kf33, m_mf33);
    arma::dvec eigval_u = arma::real(eigval_gu);
    arma::dmat eigvec_u = arma::real(eigvec_gu);

    // Sorted eigenvalues and eigenvectors
    m_eigval = arma::sort(eigval_u);
    arma::uvec eig_indices = arma::sort_index(eigval_u);
    arma::dmat eigvec = arma::zeros(eigvec_u.n_rows, eigvec_u.n_cols);

    for(uint i = 0; i < m_eigval.n_rows; i++)
    {
        eigvec.col(i) = eigvec_u.col(eig_indices(i));
    }

    // Transformation matrix initialization
    m_p_mat = arma::zeros(eigvec_u.n_rows, eigvec_u.n_cols);

    // Normalize eigenvectors
    for(uint i = 0; i < m_eigval.n_rows; i++)
    {
        double mi = arma::as_scalar((eigvec.col(i)).t() * m_mf33 * eigvec.col(i));
        m_p_mat.col(i) = (1.0 / sqrt(mi)) * eigvec.col(i);
    }
}
