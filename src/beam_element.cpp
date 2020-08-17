#include "beam_element.h"
#include "euler_rotations.h"

BeamElement::BeamElement(uint element_id, arma::ivec boundary_conditions,
    uint elements, double length, double radius, double young_modulus,
    double density)
{
    // Element id
    m_element_id = element_id;

    // Elements number
    m_elements = elements;

    // Length 
    m_length = length;

    // Radius
    m_radius = radius;

    // Young modulus 
    m_young_modulus = young_modulus;

    // Density 
    m_density = density;

    // Element weight vector 
    m_weight_F = {0.0, 0.0, - m_g * m_mfj};

    // Cross-sectional area
    m_area = M_PI * pow(m_radius, 2.0);

    // Mass 
    m_mfj = m_density * m_area * m_length;

    // Distance of element left node from reference frame
    m_raaoj_f_f = {(double)(m_element_id - 1) * m_length, 0.0, 0.0};

    // Distance of element centre of mass from reference frame
    m_racoj_f_f = { (m_length / 2.0), 0.0, 0.0 };
    m_racoj_f_f += m_raaoj_f_f;
    
    // Inertial tensor of element j wrt to the centre of mass of element j
    // (reference frame) 
    m_i_coj_f = {{m_mfj * pow(m_radius, 2.0) / 2.0, 0.0, 0.0},
        {0.0, m_mfj * (pow(m_length, 2.0) +
        3.0 * pow(m_radius, 2.0)) / 12.0, 0.0},
        {0.0, 0.0, m_mfj * (pow(m_length, 2.0) +
        3.0 * pow(m_radius, 2.0)) / 12.0}};


    // Number of dofs 
    m_total_dofs = 5 * (m_elements + 1);

    // Boundary conditions vector
    m_lb = boundary_conditions;

    // Number of dofs after bounderies
    m_dofs = m_total_dofs - m_lb.n_rows;

    // Boundary conditions matrix
    m_lb_mat = arma::zeros(m_total_dofs, m_dofs);
    arma::ivec lb_inv = arma::regspace<arma::ivec>(m_lb.back() + 1, m_total_dofs);
    for(uint i = 0; i < lb_inv.n_rows; i++) { m_lb_mat(lb_inv(i) - 1, i) = 1.0; }

    // Locator vectors for global transformation 
    arma::ivec lj_vec = arma::regspace<arma::ivec>(5 * m_element_id - 4, 1, 
        5 * m_element_id + 5);
    m_lj_mat = dm::locator_matrix(lj_vec, m_total_dofs);

    // Locator matrices for local transformation(axial and bending elements)
    m_luj_mat = dm::locator_matrix(m_luj, m_ns);
    m_lvj_mat = dm::locator_matrix(m_lvj, m_ns);
    m_lwj_mat = dm::locator_matrix(m_lwj, m_ns);

    // Caclualate shape integrals
    shape_integrals();

    //Elastic mass matrix
    elastic_mass_matrix_calculation();

    //Elastic stiffness matrix 
    elastic_stiffness_matrix_calculation();

    // Calculate stiffness matrix (constant)
    stiffness_matrix_calculation();

    // Calculate stiffness matrix (constant)
    stiffness_matrix_calculation();
}


// Update state 
void BeamElement::state_update(arma::dvec q, arma::dvec q_dot)
{
    // State
    m_roa_g_g = {q(0), q(1), q(2)};
    m_theta = {q(3), q(4), q(5)};
    m_qf = q(arma::span(6, q.n_rows - 1));

    // State dot
    m_roa_dot_g_g = {q_dot(0), q_dot(1), q_dot(2)};
    m_theta_dot = {q_dot(3), q_dot(4), q_dot(5)};
    m_qf_dot = q_dot(arma::span(6, q.n_rows - 1));
}


// Update element state, mass matrix and coriolis vector
void BeamElement::update(double t, arma::dvec q, arma::dvec q_dot) 
{
    // State update 
    state_update(q, q_dot);

    // Time update
    m_time = t;

    // G and Gdot matrices
    m_g_mat = EulerRotations::G(m_theta);
    m_g_dot_mat = EulerRotations::G_dot(m_theta, m_theta_dot);

    // Rotation matrix 
    m_rot_f_F = EulerRotations::rotation(m_theta);

    // Angular velocity (rad / sec)
    m_omega =  m_g_mat * m_theta_dot;

    // Update Nj integrals
    nj_integrals();

    // Update mass matrix 
    mass_matrix_calculation();

    // Update coriolis-centrifugal vector 
    coriolis_vector_calculation();
}


// Mass matrix calculation
void BeamElement::mass_matrix_calculation(void)
{
    // First row   
    arma::dmat mfj11 = m_mfj * arma::eye(3, 3);

    arma::dmat mfj12 = - m_rot_f_F * (m_mfj * dm::s(m_racoj_f_f) + 
        dm::s(m_nj_int[0] * m_qf)) * m_g_mat;

    arma::dmat mfj13 =  m_rot_f_F * m_nj_int[0];

    arma::dmat mfj1 = arma::join_horiz(mfj11, mfj12, mfj13);

    // Second row   
    arma::dmat mfj21 =  mfj12.t();

    arma::dmat mfj22 =  m_g_mat.t() * m_i_a_fj * m_g_mat;

    arma::dmat mfj23 = - m_g_mat.t() * (m_nj_int[4] + m_nj_int[5]);

    arma::dmat mfj2 = arma::join_horiz(mfj21, mfj22, mfj23);

    // Third row   
    arma::dmat mfj31= mfj13.t();

    arma::dmat mfj32 = mfj23.t();

    arma::dmat mfj3 = arma::join_horiz(mfj31, mfj32, m_mfj33);

    // Mass matrix
    m_mass = arma::join_vert(mfj1, mfj2, mfj3);
}


// Stiffness matrix calculation
void BeamElement::stiffness_matrix_calculation(void)
{
    // First row 
    arma::dmat kfj11 = arma::zeros(3, 3); arma::dmat kfj12 = arma::zeros(3, 3);
    arma::dmat kfj13 = arma::zeros(3, m_dofs);
    arma::dmat kfj1 = arma::join_horiz(kfj11, kfj12, kfj13);

    // Second row 
    arma::dmat kfj21 = arma::zeros(3, 3); arma::dmat kfj22 = arma::zeros(3, 3);
    arma::dmat kfj23 = arma::zeros(3, m_dofs);
    arma::dmat kfj2 = arma::join_horiz(kfj21, kfj22, kfj23);

    // Third row 
    arma::dmat kfj31 = arma::zeros(m_dofs, 3);
    arma::dmat kfj32 = arma::zeros(m_dofs, 3);
    arma::dmat kfj3 = arma::join_horiz(kfj31, kfj32, m_kfj33);

    // Total stiffness matrix 
    m_stiffness = arma::join_vert(kfj1, kfj2, kfj3);
}


// Coriolis-centrifugal vector calculation
void BeamElement::coriolis_vector_calculation(void)
{
    // First vector
    arma::dvec fvfj1 = m_rot_f_F * ((m_mfj * dm::s(m_racoj_f_f) +
        dm::s(m_nj_int[0] * m_qf)) * m_g_dot_mat * m_theta_dot -
        pow(dm::s(m_omega), 2.0) * (m_racoj_f_f * m_mfj +
        m_nj_int[0] * m_qf) - 2.0 * dm::s(m_omega) * m_nj_int[0] * m_qf_dot);
    
    // Second vector
    arma::dvec fvfj2 = m_g_mat.t() * (m_nj_int[7] +
        2.0 * (m_nj_int[8] + m_nj_int[9]) * m_qf_dot -
        m_i_a_fj * m_g_dot_mat * m_theta_dot);

    // Third vector
    arma::dvec fvfj3 = (m_nj_int[4] + m_nj_int[5]).t() *
        m_g_dot_mat * m_theta_dot - m_nj_int[10] - 2.0 * m_nj_int[11] * m_qf_dot;

    // Coriolis-Centrifugal vector
    m_fvfj = arma::join_vert(fvfj1, fvfj2, fvfj3);
}


// External force calculation
void BeamElement::calculate_external_forces(arma::dvec int1, arma::dvec int2, 
    arma::dvec int3, arma::dvec fbj_f)
{
    // First term
    arma::dvec qfj1 = m_rot_f_F * int1 + m_rot_f_F * fbj_f + 
        m_weight_F;

    // Second term
    arma::dvec qfj2 = m_g_mat.t() * (int2 + dj_x(m_length) * fbj_f + 
        dj_x(m_length / 2.0) * m_rot_f_F.t() * m_weight_F);

    // Third term
    arma::dvec qfj3 = int3 + shape_function(m_length).t() * fbj_f + 
        shape_function(m_length / 2.0).t() * m_rot_f_F.t() * m_weight_F;
    
    // Force vector 
    m_qfj = arma::join_vert(qfj1, qfj2, qfj3);
}


// Elastic mass matrix
void BeamElement::elastic_mass_matrix_calculation(void)
{
    m_mfj33 = m_phi11j_dash + m_phi22j_dash + m_phi33j_dash;
}


// Elastic stiffness matrix calculation
void BeamElement::elastic_stiffness_matrix_calculation(void)
{
    // Element area moment of inertia (m^4)
    double m_iyy = (M_PI / 4) * pow(m_radius, 4.0);
    double m_izz = (M_PI / 4) * pow(m_radius, 4.0);

    // Axial stiffness matrix
    arma::dmat ku = {{1, -1}, {-1, 1}}; 
    ku = ( (m_young_modulus * m_area) / m_length ) * ku;
    
    // Bending element y direction
    arma::dmat kv ={{12, 6 * m_length, -12, 6 * m_length},
        {6 * m_length, 4 * pow(m_length, 2.0), 
        -6 * m_length, 2 * pow(m_length, 2.0)},
        {-12, -6 * m_length, 12, -6 * m_length},
        {6 * m_length, 2 * pow(m_length, 2.0), 
        -6 * m_length, 4 * pow(m_length, 2.0)}};
    kv = ( (m_young_modulus * m_izz) / pow(m_length, 3.0) ) * kv;

    // Bending element z direction
    arma::dmat kw ={{12, 6 * m_length, -12, 6 * m_length},
        {6 * m_length, 4 * pow(m_length, 2.0), 
        -6 * m_length, 2 * pow(m_length, 2.0)},
        {-12, -6 * m_length, 12, -6 * m_length},
        {6 * m_length, 2 * pow(m_length, 2.0), 
        -6 * m_length, 4 * pow(m_length, 2.0)}};
    arma::dmat tr = {{1, -1, 1, -1}, {-1, 1, -1, 1}, {1, -1, 1, -1}, {-1, 1, -1, 1}};
    kw = ( (m_young_modulus * m_iyy) / pow(m_length, 3.0) ) * kw;
    kw = tr % kw;

    // Spatial beam stiffness matrix
    arma::dmat kej = m_luj_mat.t() * ku * m_luj_mat + m_lvj_mat.t() * kv * m_lvj_mat + 
        m_lwj_mat.t() * kw * m_lwj_mat;

    m_kfj33 = m_lb_mat.t() * m_lj_mat.t() * kej * m_lj_mat * m_lb_mat;
}


// Nj integrals calculation
void BeamElement::nj_integrals(void)
{
    /*************** Integral N1j ***************/
    arma::dmat psi_uj_int = {1.0 / 2.0, 1.0 / 2.0};
    psi_uj_int *= m_mfj;

    arma::dmat psi_vj_int = {1.0 / 2.0, m_length / 12.0, 1.0 / 2.0,
        - m_length / 12.0};
    psi_vj_int *= m_mfj;

    arma::dmat psi_wj_int = {1.0 / 2.0, - m_length / 12.0, 1.0 / 2.0,
        m_length / 12.0};
    psi_wj_int *= m_mfj;
    
    m_nj_int[0] = arma::join_vert(psi_uj_int * m_luj_mat * m_lj_mat * m_lb_mat,
        psi_vj_int * m_lvj_mat * m_lj_mat * m_lb_mat,
        psi_wj_int * m_lwj_mat * m_lj_mat * m_lb_mat);

    /*************** Integral N2j ***************/
    m_nj_int[1] = m_i_coj_f + m_mfj * dm::s(m_racoj_f_f).t() *
        dm::s(m_racoj_f_f);


    /*************** Integral N3j ***************/
    // First row
    arma::dmat n3j_row1 = arma::join_horiz((m_phi33j_hat + m_phi22j_hat) * m_qf,
        - m_phi21j_hat * m_qf, - m_phi31j_hat * m_qf);

    // Second row
    arma::dmat n3j_row2 = arma::join_horiz(- m_phi12j_hat * m_qf,
        (m_phi33j_hat + m_phi11j_hat) * m_qf, - m_phi32j_hat * m_qf);

    // Third row
    arma::dmat n3j_row3 = arma::join_horiz(- m_phi13j_hat * m_qf,
        - m_phi23j_hat * m_qf, (m_phi11j_hat + m_phi22j_hat) * m_qf);

    m_nj_int[2] = dm::s(m_raaoj_f_f).t() * dm::s(m_nj_int[0] * m_qf) *
        arma::join_vert(n3j_row1, n3j_row2, n3j_row3);


    /*************** Integral N4j ***************/
    // First row
    arma::dmat n4j_row1 = arma::join_horiz(m_qf.t() * (m_phi33j_dash +
        m_phi22j_dash) * m_qf, - m_qf.t() * m_phi21j_dash * m_qf,
        - m_qf.t() * m_phi31j_dash * m_qf);

    // Second row
    arma::dmat n4j_row2 = arma::join_horiz(- m_qf.t() * m_phi12j_dash * m_qf,
        m_qf.t() * (m_phi33j_dash + m_phi11j_dash) * m_qf,
        - m_qf.t() * m_phi32j_dash * m_qf);

    // Third row
    arma::dmat n4j_row3 = arma::join_horiz(- m_qf.t() * m_phi13j_dash * m_qf,
        - m_qf.t() * m_phi23j_dash * m_qf,
        m_qf.t() * (m_phi22j_dash + m_phi11j_dash) * m_qf);

    m_nj_int[3] = arma::join_vert(n4j_row1, n4j_row2, n4j_row3);

    /*************** Inertia update ***************/
    // Inertial tensor of element j wrt to point A (reference frame)
    m_i_a_fj = m_nj_int[1] + m_nj_int[2] + m_nj_int[2].t() + m_nj_int[3];

    /*************** Integral N5j ***************/
    m_nj_int[4] = dm::s(m_raaoj_f_f).t() * m_nj_int[0] +
        arma::join_vert(m_phi32j_hat - m_phi23j_hat, m_phi13j_hat -
        m_phi31j_hat, m_phi21j_hat - m_phi12j_hat);

    /*************** Integral N6j ***************/
    m_nj_int[5] = arma::join_vert(m_qf.t() * (m_phi32j_dash - m_phi23j_dash),
        m_qf.t() * (m_phi13j_dash - m_phi31j_dash),
        m_qf.t() * (m_phi21j_dash - m_phi12j_dash));

    /*************** Integral N7j ***************/
    m_nj_int[6] = m_phi11j_dash + m_phi22j_dash + m_phi33j_dash;

    /*************** Integral N8j ***************/
    m_nj_int[7] = - dm::s(m_omega) * (m_i_a_fj * m_omega);

    /*************** Integral N9j ***************/
    // First row
    arma::dmat n9j_row1 = - m_omega(0) * (m_phi33j_hat + m_phi22j_hat) +
        m_omega(1) * m_phi21j_hat + m_omega(2) * m_phi31j_hat;

    // Second row
    arma::dmat n9j_row2 = m_omega(0) * m_phi12j_hat - m_omega(1) * 
        (m_phi11j_hat + m_phi33j_hat) + m_omega(2) * m_phi32j_hat;

    // Third row
    arma::dmat n9j_row3 = m_omega(0) * m_phi13j_hat + m_omega(1) * m_phi23j_hat -
        m_omega(2) * (m_phi11j_hat + m_phi22j_hat);

    m_nj_int[8] = dm::s(m_raaoj_f_f).t() * dm::s(m_omega) * m_nj_int[0] +
        arma::join_vert(n9j_row1, n9j_row2, n9j_row3);

    /*************** Integral N10j ***************/
    // First row
    arma::dmat n10j_row1 = - m_qf.t() * m_omega(0) * (m_phi33j_dash +
        m_phi22j_dash) + m_qf.t() * m_omega(1) * m_phi21j_dash +
        m_qf.t() * m_omega(2) * m_phi31j_dash;

    // Second row
    arma::dmat n10j_row2 = m_qf.t() * m_omega(0) * m_phi12j_dash -
        m_qf.t() * m_omega(1) * (m_phi11j_dash + m_phi33j_dash) +
        m_qf.t() * m_omega(2) * m_phi32j_dash;

    // Third row
    arma::dmat n10j_row3 = m_qf.t() * m_omega(0) * m_phi13j_dash +
        m_qf.t() * m_omega(1) * m_phi23j_dash -
        m_qf.t() * m_omega(2) * (m_phi11j_dash + m_phi22j_dash);

    m_nj_int[9] = arma::join_vert(n10j_row1, n10j_row2, n10j_row3);

    /*************** Integral N11j ***************/
    m_nj_int[10] = (m_nj_int[8] + m_nj_int[9]).t() * m_omega;

    /*************** Integral N12j ***************/
    m_nj_int[11] = m_omega(0) * (m_phi32j_dash - m_phi23j_dash) + 
        m_omega(1) * (m_phi13j_dash - m_phi31j_dash) +
        m_omega(2) * (m_phi21j_dash - m_phi12j_dash);
}


// Shape integrals calculation 
void BeamElement::shape_integrals(void)
{
    /********* Shape integrals dash *********/
    //Integral phi11j_dash
    arma::dmat phi_uuj = {{1.0 / 3.0, 1.0 / 6.0}, {1.0 / 6.0, 1.0 / 3.0}};

    m_phi11j_dash = m_mfj * m_lb_mat.t() * m_lj_mat.t() * m_luj_mat.t() *
        phi_uuj * m_luj_mat * m_lj_mat * m_lb_mat;

    //Integral phi12j_dash
    arma::dmat phi_uvj = {{7.0 / 20.0, m_length / 20.0, 3.0 / 20.0,
        - m_length / 30.0}, {3.0 / 20.0, m_length / 30.0, 7.0 / 20.0,
        - m_length / 20.0}};

    m_phi12j_dash = m_mfj * m_lb_mat.t() * m_lj_mat.t() * m_luj_mat.t() *
        phi_uvj * m_lvj_mat * m_lj_mat * m_lb_mat;

    //Integral phi13j_dash
    arma::dmat phi_uwj = {{7.0 / 20.0, - m_length / 20.0, 3.0 / 20.0,
        m_length / 30.0}, {3.0 / 20.0, - m_length / 30.0, 7.0 / 20.0,
        m_length / 20.0}};

    m_phi13j_dash = m_mfj * m_lb_mat.t() * m_lj_mat.t() * m_luj_mat.t() *
        phi_uwj * m_lwj_mat * m_lj_mat * m_lb_mat;

    //Integral phi21j_dash
    m_phi21j_dash = m_phi12j_dash.t();

    //Integral phi22j_dash
    arma::dmat phi_vvj = {{156.0, 22.0 * m_length, 54.0, - 13.0 * m_length}, 
        {22.0 * m_length, 4.0 * pow(m_length, 2.0), 13.0 * m_length, 
        - 3.0 * pow(m_length, 2.0)},
        {54.0, 13.0 * m_length, 156.0, - 22.0 * m_length}, 
        {- 13.0 * m_length, - 3.0 * pow(m_length, 2.0), 
        - 22.0 * m_length, 4.0 * pow(m_length, 2.0)}};

    m_phi22j_dash = (m_mfj / 420.0) * m_lb_mat.t() * m_lj_mat.t() *
        m_lvj_mat.t() * phi_vvj * m_lvj_mat * m_lj_mat * m_lb_mat;

    //Integral phi23j_dash
    arma::dmat phi_vwj = {{156.0, - 22.0 * m_length, 54.0, 13.0 * m_length}, 
        {22.0 * m_length, - 4.0 * pow(m_length, 2.0), 13.0 * m_length, 
        3.0 * pow(m_length, 2.0)}, 
        {54.0, - 13.0 * m_length, 156.0, 22.0 * m_length}, 
        {- 13.0 * m_length, 3.0 * pow(m_length, 2.0), 
        - 22.0 * m_length, - 4.0 * pow(m_length, 2.0)}}; 

    m_phi23j_dash = (m_mfj / 420.0) * m_lb_mat.t() * m_lj_mat.t() *
        m_lvj_mat.t() * phi_vwj * m_lwj_mat * m_lj_mat * m_lb_mat;

    //Integral phi31j_dash
    m_phi31j_dash = m_phi13j_dash.t();

    //Integral phi32j_dash
    m_phi32j_dash = m_phi23j_dash.t();

    //Integral phi33j_dash
    arma::dmat phi_wwj = {{156.0, - 22.0 * m_length, 54.0, 13.0 * m_length}, 
        {- 22.0 * m_length, 4.0 * pow(m_length, 2.0), - 13.0 * m_length, 
        - 3.0 * pow(m_length, 2.0)}, 
        {54.0, - 13.0 * m_length, 156.0, 22.0 * m_length}, 
        {13.0 * m_length, - 3.0 * pow(m_length, 2.0), 
        22.0 * m_length, 4.0 * pow(m_length, 2.0)}};

    m_phi33j_dash = (m_mfj / 420.0) * m_lb_mat.t() * m_lj_mat.t() * m_lwj_mat.t() *
        phi_wwj * m_lwj_mat * m_lj_mat * m_lb_mat;

    /********* Shape integrals hat *********/
    //Integral phi11j_hat
    arma::drowvec phi_x1uj = {1.0 / 6.0, 1.0 / 3.0};
    m_phi11j_hat  = m_mfj * m_length * phi_x1uj * m_luj_mat * m_lj_mat * m_lb_mat;                          

    //Integral phi12j_hat
    arma::drowvec phi_x1vj = {3.0 / 20.0, m_length / 30.0, 7.0 / 20.0, 
        - m_length / 20.0};
    m_phi12j_hat  = m_mfj * m_length * phi_x1vj * m_lvj_mat * m_lj_mat * m_lb_mat;                          

    //Integral phi13j_hat
    arma::drowvec phi_x1wj = {3.0 / 20.0, - m_length / 30.0, 7.0 / 20.0, 
        m_length / 20.0};
    m_phi13j_hat  = m_mfj * m_length * phi_x1wj * m_lwj_mat * m_lj_mat * m_lb_mat;                          

    //Integral phi21j_hat
    arma::drowvec phi_x2uj = arma::zeros<arma::drowvec>(1, m_axial_dofs);
    m_phi21j_hat = phi_x2uj * m_luj_mat * m_lj_mat * m_lb_mat;

    //Integral phi22j_hat
    arma::drowvec phi_x2vj = arma::zeros<arma::drowvec>(1, m_bending_y_dofs);
    m_phi22j_hat = phi_x2vj * m_lvj_mat * m_lj_mat * m_lb_mat;

    //Integral phi23j_hat
    arma::drowvec phi_x2wj = arma::zeros<arma::drowvec>(1, m_bending_z_dofs);
    m_phi23j_hat = phi_x2wj * m_lwj_mat * m_lj_mat * m_lb_mat;

    //Integral phi31j_hat
    arma::drowvec phi_x3uj = arma::zeros<arma::drowvec>(1, m_axial_dofs);
    m_phi31j_hat = phi_x3uj * m_luj_mat * m_lj_mat * m_lb_mat;

    //Integral phi32j_hat
    arma::drowvec phi_x3vj = arma::zeros<arma::drowvec>(1, m_bending_y_dofs);
    m_phi32j_hat = phi_x3vj * m_lvj_mat * m_lj_mat * m_lb_mat;

    //Integral phi33j_hat
    arma::drowvec phi_x3wj = arma::zeros<arma::drowvec>(1, m_bending_z_dofs);
    m_phi33j_hat = phi_x3wj * m_lwj_mat * m_lj_mat * m_lb_mat;
}


// Shape function 
arma::dmat BeamElement::shape_function(double xj)
{
    // Dimensionless lenght
    double ksi = xj / m_length;

    // Shape function x direction
    arma::drowvec psi_u = {1 - ksi, ksi};

    // Shape function y direction
    arma::drowvec psi_v = {1 - 3 * pow(ksi, 2.0)  + 2 * pow(ksi, 3.0),
        m_length * (ksi - 2 * pow(ksi, 2.0) + pow(ksi, 3.0)), 
        3 * pow(ksi, 2.0) - 2 * pow(ksi, 3.0), 
        m_length * (pow(ksi, 3.0) - pow(ksi, 2.0))};

    // Shape function z direction
    arma::drowvec psi_w = {1 - 3 * pow(ksi, 2.0)  + 2 * pow(ksi, 3.0),
        - m_length * (ksi - 2 * pow(ksi, 2.0) + pow(ksi, 3.0)), 
        3 * pow(ksi, 2.0) - 2 * pow(ksi, 3.0), 
        - m_length * (pow(ksi, 3.0) - pow(ksi, 2.0))};

    // Total shape function 
    return arma::join_vert(psi_u * m_luj_mat * m_lj_mat * m_lb_mat,
        psi_v * m_lvj_mat * m_lj_mat * m_lb_mat,
        psi_w * m_lwj_mat * m_lj_mat * m_lb_mat);
}


// Dj(x) function
arma::dmat BeamElement::dj_x(double xj)
{
    // Central line distance 
    arma::dvec raojpocj_f_f = {xj, 0.0, 0.0};

    arma::dmat dj_matrix = dm::s(m_raaoj_f_f + raojpocj_f_f
        + shape_function(xj) * m_qf);

    return dj_matrix;
}


// Get deflection with respect to f frame
arma::dvec BeamElement::get_deflection(double xj, arma::dvec elastic_coordinates)
{
    return shape_function(xj) * elastic_coordinates;
}