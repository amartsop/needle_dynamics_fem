#include "beam_element.h"
#include "euler_rotations.h"

BeamElement::BeamElement(uint element_id, uint elements, double length, double radius, 
        double young_modulus, double density)
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

    // Cross-sectional area
    m_area = M_PI * pow(m_radius, 2.0);

    // Distance of element centre of mass from reference frame
    m_racoj_f_f = {(double)(m_element_id - 1) * m_length + (m_length / 2.0), 0.0,
        0.0 };
                                          
    // Distance of element centre of mass from reference frame
    m_raboj_f_f = {(double)(m_element_id - 1) * m_length + (m_length), 0.0,
        0.0 };
                                          
    // Number of dofs 
    m_dofs = 5 * (m_elements + 1);

    // Locator vectors for global transformation 
    arma::ivec lj_vec = arma::regspace<arma::ivec>(5 * m_element_id - 4, 1, 
        5 * m_element_id + 5);

    m_lj_mat = dm::locator_matrix(lj_vec, m_dofs);

    // Locator matrices for local transformation(axial and bending elements)
    m_luj_mat = dm::locator_matrix(m_luj, m_ns);
    m_lvj_mat = dm::locator_matrix(m_lvj, m_ns);
    m_lwj_mat = dm::locator_matrix(m_lwj, m_ns);

    // Mass 
    m_mfj = m_density * m_area * m_length;

    // Element weight vector 
    m_weight_F = {0.0, 0.0, - m_g * m_mfj};

    // Caclualate shape integrals
    shape_integrals_calculation();

    // Calculate stiffness matrix
    stiffness_matrix_calculation();
}


void BeamElement::update(arma::dvec q, arma::dvec q_dot, arma::dvec fbj_f, 
    arma::dvec tsj_f)
{
    // Update mass matrix
    mass_matrix_calculation(q, q_dot);
    
    // Update coriolis-centrifugal vector
    coriolis_vector_calculation(q, q_dot);

    // Update external force vector
    external_force_vector_calculation(q, q_dot, fbj_f, tsj_f);
}


void BeamElement::mass_matrix_calculation(arma::dvec q, arma::dvec q_dot) 
{
    // State 
    arma::dvec theta = state::theta(q, q_dot);
    arma::dvec qf = state::qf(q, q_dot);

    // Rotation and G matrix
    arma::dmat rot_f_F = EulerRotations::rotation(theta);
    arma::dmat g_mat = EulerRotations::G(theta);

    // Inertial tensor
    arma::dmat ifj = inertial_tensor_calculation(q, q_dot);

    // Gammaj matrix 
    arma::dmat gammaj_mat =
        arma::join_vert(qf.t() * m_lj_mat.t() * (m_phi32j - m_phi23j), 
        qf.t() * m_lj_mat.t() * (m_phi13j - m_phi31j),
        qf.t() * m_lj_mat.t() * (m_phi21j - m_phi12j));

    // First row
    arma::dmat mfj11 = m_mfj * arma::eye(3, 3);
    arma::dmat mfj12 = - rot_f_F * (m_mfj * dm::s(m_racoj_f_f) + 
        dm::s(m_nj * m_lj_mat * qf)) * g_mat;
    arma::dmat mfj13 = rot_f_F * m_nj * m_lj_mat;
    arma::dmat mfj1 = arma::join_horiz(mfj11, mfj12, mfj13);

    // Second row 
    arma::dmat mfj21 = mfj12.t();
    arma::dmat mfj22 = g_mat.t() * ifj * g_mat;
    arma::dmat mfj23 = - g_mat * gammaj_mat * m_lj_mat;
    arma::dmat mfj2 = arma::join_horiz(mfj21, mfj22, mfj23);

    // Third row 
    arma::dmat mfj31 = mfj13.t();
    arma::dmat mfj32 = mfj23.t();
    arma::dmat mfj33 = m_lj_mat.t() * (m_phi11j + m_phi22j + m_phi33j) * m_lj_mat;
    arma::dmat mfj3 = arma::join_horiz(mfj31, mfj32, mfj33);

    // Mass matrix
    m_mass = arma::join_vert(mfj1, mfj2, mfj3);
}


void BeamElement::stiffness_matrix_calculation(void)
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

    // First row
    arma::dmat kfj1 = arma::join_horiz(arma::zeros(3, 3), arma::zeros(3, 3), 
        arma::zeros(3, m_dofs));

    // Third rows 
    arma::dmat kfj3 = arma::join_horiz(arma::zeros(m_dofs, 3),
        arma::zeros(m_dofs, 3), m_lj_mat.t() * kej * m_lj_mat);

    // Total stiffness matrix 
    m_stiffness = arma::join_vert(kfj1, kfj1, kfj3);
}


void BeamElement::coriolis_vector_calculation(arma::dvec q, arma::dvec q_dot)
{
    // State 
    arma::dvec theta = state::theta(q, q_dot);
    arma::dvec qf = state::qf(q, q_dot);

    // State dot
    arma::dvec theta_dot = state::theta_dot(q, q_dot);
    arma::dvec qf_dot = state::qf_dot(q, q_dot);

    // Rotation, G and Gdot matrix
    arma::dmat rot_f_F = EulerRotations::rotation(theta);
    arma::dmat g_mat = EulerRotations::G(theta);
    arma::dmat g_dot_mat = EulerRotations::G_dot(theta, theta_dot);

    // Angular velocity (rad / sec)
    arma::dvec omega =  g_mat * theta_dot;

    // Inertial tensor
    arma::dmat ifj = inertial_tensor_calculation(q, q_dot);

    // Dj matrix 
    arma::mat dj_mat = dj_matrix_caclulation(q, q_dot);

    // Gammaj matrix 
    arma::dmat gammaj_mat =
        arma::join_vert(qf.t() * m_lj_mat.t() * (m_phi32j - m_phi23j), 
        qf.t() * m_lj_mat.t() * (m_phi13j - m_phi31j),
        qf.t() * m_lj_mat.t() * (m_phi21j - m_phi12j));

    // Aj matrix 
    arma::dmat aj_mat = omega(0) * (m_phi32j - m_phi23j) + 
        omega(1) * (m_phi13j - m_phi31j) + omega(2) * (m_phi21j - m_phi12j);
    
    // First vector
    arma::dvec fvfj1 = - rot_f_F * dm::s(omega) * dm::s(omega) * 
        (m_racoj_f_f * m_mfj + m_nj * m_lj_mat * qf) + rot_f_F * (m_mfj * 
        dm::s(m_racoj_f_f) + dm::s(m_nj * m_lj_mat * qf)) * g_dot_mat * 
        theta_dot - 2 * rot_f_F * dm::s(omega) * m_nj * m_lj_mat * qf_dot;

    // Second vector
    arma::dvec fvfj2 = g_mat.t() * ( 2.0 * dj_mat * m_lj_mat * qf_dot 
        - ifj * g_dot_mat * theta_dot - dm::s(omega) * (ifj * omega) );

    // Third vector
    arma::dvec fvfj3 = m_lj_mat.t() * (dj_mat.t() * omega +
        gammaj_mat.t() * g_dot_mat * theta_dot - 2.0 * aj_mat * m_lj_mat * qf_dot );

    // Coriolis-Centrifugal vector
    m_fvfj = arma::join_vert(fvfj1, fvfj2, fvfj3);
}

void BeamElement::external_force_vector_calculation(arma::dvec q, arma::dvec q_dot,
    arma::dvec fbj_f, arma::dvec tsj_f)
{
    // State 
    arma::dvec theta = state::theta(q, q_dot);
    arma::dvec qf = state::qf(q, q_dot);

    // Rotation, G and Gdot matrix
    arma::dmat rot_f_F = EulerRotations::rotation(theta);
    arma::dmat g_mat = EulerRotations::G(theta);

    // First vector
    arma::dvec qfj1 = rot_f_F * fbj_f + m_weight_F +
        (2.0 * M_PI * m_radius * m_length) * rot_f_F * tsj_f;

    // Second vector
    arma::dvec muj_vec = {m_radius * M_PI * pow(m_length, 2.0), 0.0, 0.0};
    arma::dmat oj_mat = oj_matrix_caclulation();
    arma::dvec qfj2 = g_mat.t() * ( dm::s(m_raboj_f_f + shape_function(1) * 
        m_lj_mat * qf) * fbj_f + dm::s(m_racoj_f_f + shape_function(0.5) * 
        m_lj_mat * qf) * rot_f_F.t() * m_weight_F + dm::s(muj_vec + oj_mat * 
        m_lj_mat * qf) * tsj_f);

    // Third vector
    arma::dvec qfj3 = m_lj_mat.t() * ( shape_function(1).t() * fbj_f + 
        shape_function(0.5).t() * rot_f_F.t() * m_weight_F + oj_mat.t() * tsj_f );

    // Force vector 
    m_qfj = arma::join_vert(qfj1, qfj2, qfj3);
}


arma::dmat BeamElement::inertial_tensor_calculation(arma::dvec q, arma::dvec q_dot)
{
    // State
    arma::dvec qf = state::qf(q, q_dot);
    
    // First row
    arma::dmat i11fj = qf.t() * m_lj_mat.t() * (m_phi22j + m_phi33j) * 
        m_lj_mat * qf;
    arma::dmat i12fj = - qf.t() * m_lj_mat.t() * (m_phi12j) * m_lj_mat * qf;
    arma::dmat i13fj = - qf.t() * m_lj_mat.t() * (m_phi13j) * m_lj_mat * qf;
    arma::dmat i1fj = arma::join_horiz(i11fj, i12fj, i13fj);

    // Second row
    arma::dmat i21fj = - qf.t() * m_lj_mat.t() * (m_phi21j) * m_lj_mat * qf;
    arma::dmat i22fj = qf.t() * m_lj_mat.t() * (m_phi11j + m_phi33j) * 
        m_lj_mat * qf;
    arma::dmat i23fj = - qf.t() * m_lj_mat.t() * (m_phi23j) * m_lj_mat * qf;
    arma::dmat i2fj = arma::join_horiz(i21fj, i22fj, i23fj);

    // Third row
    arma::dmat i31fj = - qf.t() * m_lj_mat.t() * (m_phi31j) * m_lj_mat * qf;
    arma::dmat i32fj = - qf.t() * m_lj_mat.t() * (m_phi32j) * m_lj_mat * qf;
    arma::dmat i33fj = qf.t() * m_lj_mat.t() * (m_phi11j + m_phi22j) * 
        m_lj_mat * qf;
    arma::dmat i3fj = arma::join_horiz(i31fj, i32fj, i33fj);

    // Inertia matrix
    return arma::join_vert(i1fj, i2fj, i3fj);
}


arma::dmat BeamElement::dj_matrix_caclulation(arma::dvec q, arma::dvec q_dot)
{

    // State 
    arma::dvec theta = state::theta(q, q_dot);
    arma::dvec qf = state::qf(q, q_dot);
    arma::dvec theta_dot = state::theta_dot(q, q_dot);

    // Rotation and G matrix
    arma::dmat g_mat = EulerRotations::G(theta);

    // Angular velocity (rad / sec)
    arma::dvec omega =  g_mat * theta_dot;

    // First row
    arma::mat dj_mat1 = qf.t() * m_lj_mat.t() * ( - (m_phi22j + m_phi33j) * 
        omega(0) + m_phi21j * omega(1) + m_phi31j * omega(2) );

    // Second row 
    arma::mat dj_mat2 = qf.t() * m_lj_mat.t() * ( m_phi12j * omega(0) - 
        (m_phi11j + m_phi33j) * omega(1) + m_phi32j * omega(2) );

    // Third row 
    arma::mat dj_mat3 = qf.t() * m_lj_mat.t() * ( m_phi13j * omega(0) + 
        m_phi23j * omega(1) - (m_phi11j + m_phi22j) * omega(2) );

    return arma::join_vert(dj_mat1, dj_mat2, dj_mat3);
}

arma::dmat BeamElement::oj_matrix_caclulation(void)
{

    // Phiu integral with respect to area
    arma::drowvec phi_u = {0.5, 0.5};
    phi_u *= (2 * M_PI * m_radius * m_length);

    // Phiv integral with respect to area
    arma::drowvec phi_v = {0.5, m_length / 12.0, 0.5, - m_length / 12.0};
    phi_v *=  (2 * M_PI * m_radius * m_length);

    // Phiw integral with respect to area
    arma::drowvec phi_w = {0.5, - m_length / 12.0, 0.5, m_length / 12.0};
    phi_w *=  (2 * M_PI * m_radius * m_length);

    return arma::join_vert(phi_u * m_luj_mat, phi_v * m_lvj_mat, phi_w * m_lwj_mat);
}

void BeamElement::shape_integrals_calculation(void)
{
    // Shape functions per direction integrals
    arma::drowvec phiuj_integ =  {0.5, 0.5}; 
    phiuj_integ *= m_mfj;
                                          
    arma::drowvec phivj_integ = {0.5, m_length / 12.0, 0.5, - m_length / 12.0};
    phivj_integ *= m_mfj;

    arma::drowvec phiwj_integ = {0.5, - m_length / 12.0, 0.5, m_length / 12.0};
    phiwj_integ *= m_mfj;

    // Nj integral
    m_nj = arma::join_vert(phiuj_integ * m_luj_mat, phivj_integ * m_lvj_mat, 
        phiwj_integ * m_lwj_mat);
    
    // Phi11j integral
    m_phi11j = {{1.0 / 3.0, 1.0 / 6.0}, {1.0 / 6.0, 1.0 / 3.0}};
    m_phi11j = m_mfj * m_luj_mat.t() * m_phi11j * m_luj_mat;

    // Phi12j integral
    m_phi12j = {{7.0 / 20.0, m_length / 20.0, 3.0 / 20.0, - m_length / 30.0},
        {3.0 / 20.0, m_length / 30.0, 7.0 / 20.0, - m_length / 20.0}};
    m_phi12j = m_mfj * m_luj_mat.t() * m_phi12j * m_lvj_mat;

    // Phi13j integral
    m_phi13j = {{7.0 / 20.0, - m_length / 20.0, 3.0 / 20.0, m_length / 30.0},
        {3.0 / 20.0, - m_length / 30.0, 7.0 / 20.0, m_length / 20.0}};
    m_phi13j = m_mfj * m_luj_mat.t() * m_phi13j * m_lwj_mat;

    // Phi21j integral 
    m_phi21j = m_phi12j.t();

    // Phi22j integral 
    m_phi22j = {{156.0, 22.0 * m_length, 54.0, - 13.0 * m_length}, 
        {22.0 * m_length, 4.0 * pow(m_length, 2.0), 13.0 * m_length, 
        - 3.0 * pow(m_length, 2.0)},
        {54.0, 13.0 * m_length, 156.0, - 22.0 * m_length}, 
        {- 13.0 * m_length, - 3.0 * pow(m_length, 2.0), 
        - 22.0 * m_length, 4.0 * pow(m_length, 2.0)}};
    m_phi22j = (m_mfj / 420.0) * m_lvj_mat.t() * m_phi22j * m_lvj_mat;

    // Phi23j integral 
    m_phi23j = {{156.0, - 22.0 * m_length, 54.0, 13.0 * m_length}, 
        {22.0 * m_length, - 4.0 * pow(m_length, 2.0), 13.0 * m_length, 
        3.0 * pow(m_length, 2.0)}, 
        {54.0, - 13.0 * m_length, 156.0, 22.0 * m_length}, 
        {- 13.0 * m_length, 3.0 * pow(m_length, 2.0), 
        - 22.0 * m_length, - 4.0 * pow(m_length, 2.0)}};
    m_phi23j = (m_mfj / 420.0) * m_lvj_mat.t() * m_phi23j * m_lwj_mat;

    // Phi31j integral 
    m_phi31j = m_phi13j.t();

    // Phi32j integral 
    m_phi32j = m_phi23j.t();

    // Phi33j integral
    m_phi33j = {{156.0, - 22.0 * m_length, 54.0, 13.0 * m_length}, 
        {- 22.0 * m_length, 4.0 * pow(m_length, 2.0), - 13.0 * m_length, 
        - 3.0 * pow(m_length, 2.0)}, 
        {54.0, - 13.0 * m_length, 156.0, 22.0 * m_length}, 
        {13.0 * m_length, - 3.0 * pow(m_length, 2.0), 
        22.0 * m_length, 4.0 * pow(m_length, 2.0)}};
    m_phi33j = (m_mfj / 420.0) * m_lwj_mat.t() * m_phi33j * m_lwj_mat;
}

arma::dmat BeamElement::shape_function(double ksi)
{

    // Shape function x direction
    arma::drowvec phi_u = {1 - ksi, ksi};

    // Shape function y direction
    arma::drowvec phi_v = {1 - 3 * pow(ksi, 2.0)  + 2 * pow(ksi, 3.0),
        m_length * (ksi - 2 * pow(ksi, 2.0) + pow(ksi, 3.0)), 
        3 * pow(ksi, 2.0) - 2 * pow(ksi, 3.0), 
        m_length * (pow(ksi, 3.0) - pow(ksi, 2.0))};

    // Shape function z direction
    arma::drowvec phi_w = {1 - 3 * pow(ksi, 2.0)  + 2 * pow(ksi, 3.0),
        - m_length * (ksi - 2 * pow(ksi, 2.0) + pow(ksi, 3.0)), 
        3 * pow(ksi, 2.0) - 2 * pow(ksi, 3.0), 
        - m_length * (pow(ksi, 3.0) - pow(ksi, 2.0))};

    // Total shape function 
    return arma::join_vert(phi_u * m_luj_mat, phi_v * m_lvj_mat,
        phi_w * m_lwj_mat);
}


arma::dvec BeamElement::get_deflection(double ksi, arma::dvec ej)
{
    return shape_function(ksi) * ej;
}