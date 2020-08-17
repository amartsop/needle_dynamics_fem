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
    arma::dmat mf33 = m_needle_ptr->get_mf33_matrix();

    // Stiffness matrices 
    arma::dmat kf33 = m_needle_ptr->get_kf33_matrix();

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

    // // Calculate elastic acceleration
    // arma::dvec qf_ddot = arma::solve(mf33, tau3 - cf33 * qf_dot - kf33 * qf);

    // // Total Acceleration
    // arma::dvec q_ddot = arma::join_vert(roa_ddot_F_F, theta_ddot_f, qf_ddot);

    // // Reaction forces 
    // arma::dmat mf = m_needle_ptr->get_mass_matrix();
    // arma::dmat cf = m_needle_ptr->get_damping_matrix();
    // arma::dmat kf = m_needle_ptr->get_stiffness_matrix();
    // arma::dvec fvf = m_needle_ptr->get_coriolis_vector();
    // arma::dvec qforce = m_needle_ptr->get_external_force();

    // // Total reaction force
    // arma::dvec reaction_force = mf * q_ddot + cf * q_dot + kf * q - fvf - qforce;

    // // Reaction force at point a 
    // arma::dvec qc1 = {reaction_force(0), reaction_force(1), reaction_force(2)};
    // arma::dvec qc2 = {reaction_force(3), reaction_force(4), reaction_force(5)};
    // arma::dvec fa_F = - qc1; arma::dvec ma_f = - arma::solve(g_mat.t(), qc2);

    // // Reaction force at point c
    // arma::dvec fc_F = m_handle_ptr->get_handle_mass() * roc_ddot_F_F -
    //     m_handle_ptr->get_weight_force() - fa_F;

    // m_fc_f = rot_f_F.t() * fc_F;

    // // Handle inertial tensor 
    // arma::dmat ic_f = m_handle_ptr->get_inertial_tensor(); 

    // // Reaction moment at point c
    // m_mc_f = ic_f * (g_dot_mat * theta_r + g_mat * theta_ddot_r) - ma_f -
    //     dm::s(m_handle_ptr->get_rca_f_f()) * rot_f_F.t() * fa_F -
    //     dm::s(omega) * ic_f * omega;


    // return arma::join_vert(qf_dot, qf_ddot);



}


// void SystemFem::update(double t, arma::dvec q, arma::dvec q_dot)
// {
//     // Update mass, stiffness, coriolis and external forces 
//     m_handle_ptr->update(t, q, q_dot);
//     m_needle_ptr->update(t, q, q_dot);

//     // Total mass matrix
//     m_mass = m_handle_ptr->get_mass_matrix() + m_needle_ptr->get_mass_matrix();

//     // Total stiffness matrix 
//     m_stiffness = m_needle_ptr->get_stiffness_matrix();

//     // Total coriolis force matrix 
//     m_fv = m_handle_ptr->get_coriolis_vector() + m_needle_ptr->get_coriolis_vector();
// }

// Calculate coordinates transformation matrix 
void SystemFem::coordinate_transformation_matrix(void)
{
    // Unsorted eigevalues and eigenvectors
    arma::cx_vec eigval_gu; arma::cx_mat eigvec_gu;
    arma::eig_pair(eigval_gu, eigvec_gu, m_kf33, m_mf33);
    arma::dvec eigval_u = arma::real(eigval_gu);
    arma::dmat eigvec_u = arma::real(eigvec_gu);

    // Sorted eigenvalues and eigenvectors
    arma::dvec eigval = arma::sort(eigval_u);
    arma::uvec eig_indices = arma::sort_index(eigval_u);
    arma::dmat eigvec = arma::zeros(eigvec_u.n_rows, eigvec_u.n_cols);

    for(uint i = 0; i < eigval.n_rows; i++)
    {
        eigvec.col(i) = eigvec_u.col(eig_indices(i));
    }

    // Transformation matrix initialization
    m_p_mat = arma::zeros(eigvec_u.n_rows, eigvec_u.n_cols);

    // Normalize eigenvectors
    for(uint i = 0; i < eigval.n_rows; i++)
    {
        double mi = arma::as_scalar((eigvec.col(i)).t() * m_mf33 * eigvec.col(i));
        m_p_mat.col(i) = (1.0 / sqrt(mi)) * eigvec.col(i);
    }
}

