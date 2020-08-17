#pragma once

#include <iostream>
#include <armadillo>
#include "dynamics_math.h"

class BeamElement
{
public:
    // Constructor
    BeamElement(uint element_id, arma::ivec boundary_conditions,
        uint elements, double length, double radius, double young_modulus,
        double density);

    // Update element state, mass matrix and coriolis vector
    void update(double t, arma::dvec q, arma::dvec q_dot);

    // Mass matrix getter
    arma::dmat get_mass_matrix(void){ return m_mass; }
    
    // Stiffness matrix getter
    arma::dmat get_stiffness_matrix(void){ return m_stiffness; }

    // Coriolis vector getter
    arma::dvec get_coriolis_vector(void){ return m_fvfj; }

    // External force vector getter
    arma::dvec get_external_force_vector(void){ return m_qfj; }

    // Get the element id
    uint get_element_id(void) { return m_element_id; }

    // Get number of dofs 
    uint get_dofs_number(void) { return m_total_dofs; }

    // External forces 
    void calculate_external_forces(arma::dvec int1, arma::dvec int2, 
        arma::dvec int3, arma::dvec fbj_f);

    // Get element's deflection
    arma::dvec get_deflection(double xj, arma::dvec elastic_coordinates);

private:
    // Element id
    uint m_element_id;

    // Elements number
    uint m_elements;

    // Length (m)
    double m_length;

    // Radius (m)
    double m_radius;

    // Young modulus (N / m^2)
    double m_young_modulus;

    // Density (kg / m^2)
    double m_density;

    // Element cross-sectional area (m^2)
    double m_area;

    // Distance of element left node from reference frame (m)
    arma::dvec m_raaoj_f_f;

    // Distance of element centre of mass from reference frame (m)
    arma::dvec m_racoj_f_f;

    // Inertial tensor of element j wrt to the centre of mass
    // of element j(reference frame) (kg * m^2)
    arma::dmat m_i_coj_f;

    // Inertial tensor of element j wrt to point A (reference frame) (kg * m^2)
    arma::dmat m_i_a_fj;

    // Number of total dofs
    uint m_total_dofs;

    // Number of dofs after boundaries
    uint m_dofs;

    // Mass (kg)
    double m_mfj;

    // Gravity constant (m / s^2)
    const double m_g = 9.80665 ;

    // Element weight vector (N)
    arma::dvec m_weight_F;

private:
    // State 
    arma::dvec m_roa_g_g, m_theta, m_qf, m_omega;

    // State dot 
    arma::dvec m_roa_dot_g_g, m_theta_dot, m_qf_dot;

    // G and Gdot matrices
    arma::dmat m_g_mat, m_g_dot_mat;

    // Rotation matrix 
    arma::dmat m_rot_f_F;

    // Current time 
    double m_time;

private:
    // Shape integrals dash
    arma::dmat m_phi11j_dash, m_phi12j_dash, m_phi13j_dash;
    arma::dmat m_phi21j_dash, m_phi22j_dash, m_phi23j_dash;
    arma::dmat m_phi31j_dash, m_phi32j_dash, m_phi33j_dash;

    // Shape integrals hat
    arma::drowvec m_phi11j_hat, m_phi12j_hat, m_phi13j_hat;
    arma::drowvec m_phi21j_hat, m_phi22j_hat, m_phi23j_hat;
    arma::drowvec m_phi31j_hat, m_phi32j_hat, m_phi33j_hat;

    // Nj integrals
    arma::dmat m_nj_int[12];
    
    // Number of axial dofs 
    const uint m_axial_dofs = 2;

    // Number of bending dofs y direction
    const uint m_bending_y_dofs = 4;

    // Number of bending dofs z direction
    const uint m_bending_z_dofs = 4;

    // Locator vectors
    const arma::ivec m_luj = {1, 6};
    const arma::ivec m_lvj = {2, 4, 7, 9};
    const arma::ivec m_lwj = {3, 5, 8, 10}; 

    // Locator matrices 
    arma::dmat m_luj_mat, m_lvj_mat, m_lwj_mat;

    // Number of dofs per element
    static const int m_ns = 10;

    // Locator matrix for global transformation
    arma::dmat m_lj_mat;

    // Boundary conditions vector
    arma::ivec m_lb;

    // Boundary conditions matrix 
    arma::dmat m_lb_mat;

private:
    // Element mass matrix 
    arma::dmat m_mass;

    // Element stiffness matrix 
    arma::dmat m_stiffness;

    // Coriolis-Centrifugal vector
    arma::dvec m_fvfj;

    // External force vector
    arma::dvec m_qfj;

private:

    // State update 
    void state_update(arma::dvec q, arma::dvec q_dot);

    // Mass matrix calculation
    void mass_matrix_calculation(void); 

    // Stiffness matrix calculation
    void stiffness_matrix_calculation(void);

    // Coriolis-Centrifugal calculation
    void coriolis_vector_calculation(void);

    // Nj integrals 
    void nj_integrals(void);

    // Shape integrals calculation
    void shape_integrals(void);

public:
    arma::dmat get_mfj33_matrix(void){ return m_mfj33; }
    arma::dmat get_kfj33_matrix(void){ return m_kfj33; }

private:
    // Mass and stiffness matrix elastic components 
    arma::dmat m_mfj33, m_kfj33;

    // Elastic mass matrix calculation (constant)
    void elastic_mass_matrix_calculation(void);

    // Elastic stiffness matrix calculation
    void elastic_stiffness_matrix_calculation(void);

public:
    // Shape function calculation
    arma::dmat shape_function(double xj);

    // Dj(x) function
    arma::dmat dj_x(double xj);

};