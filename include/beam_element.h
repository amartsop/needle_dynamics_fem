#pragma once

#include <iostream>
#include <armadillo>
#include "dynamics_math.h"
#include "state.h"

class BeamElement
{
public:
    // Constructor
    BeamElement(uint element_id, uint elements, double length, double radius, 
        double young_modulus, double density);

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

    // Set element id 
    void set_element_id(uint element_id); 

    // Get number of dofs 
    uint get_dofs_number(void) { return m_dofs; }

    // Update element matrices, coriolis and external force vector
    void update(arma::dvec q, arma::dvec q_dot, arma::dvec fbj_f, 
        arma::dvec tsj_f);

    // Get element's deflection
    arma::dvec get_deflection(double ksi, arma::dvec ej);

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

    // Distance of element centre of mass from reference frame (m)
    arma::dvec m_racoj_f_f;

    // Distance of element right node from reference frame (m)
    arma::dvec m_raboj_f_f;

    // Number of dofs
    uint m_dofs;

    // Mass (kg)
    double m_mfj;

    // Gravity constant (m / s^2)
    const double m_g = 9.80665 ;

    // Element weight vector (N)
    arma::dvec m_weight_F;

private:
    // Shape integrals
    arma::dmat m_nj, m_phi11j, m_phi12j, m_phi13j, m_phi22j, m_phi23j, m_phi33j;
    arma::dmat m_phi21j, m_phi31j, m_phi32j;

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

    // Inertial tensor calculation
    arma::dmat inertial_tensor_calculation(arma::dvec q, arma::dvec q_dot);

    // Dj matrix 
    arma::dmat dj_matrix_caclulation(arma::dvec q, arma::dvec q_dot);

    // Oj matrix 
    arma::dmat oj_matrix_caclulation(void);

    // Mass matrix calculation
    void mass_matrix_calculation(arma::dvec q, arma::dvec q_dot); 

    // Stiffness matrix calculation
    void stiffness_matrix_calculation(void);

    // Coriolis-Centrifugal calculation
    void coriolis_vector_calculation(arma::dvec q, arma::dvec q_dot);

    // External force vector calculation
    void external_force_vector_calculation(arma::dvec q, arma::dvec q_dot,
        arma::dvec fbj_f, arma::dvec tsj_f);

    // Shape integrals calculation
    void shape_integrals_calculation(void);

    // Shape function calculation
    arma::dmat shape_function(double ksi);

};