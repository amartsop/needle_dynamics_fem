#pragma once

#include <iostream>

#include <iostream>
#include <armadillo>
#include <vector>

#include "state.h"
#include "needle_properties.hpp"
#include "dynamics_math.h"
#include "beam_element.h"

class FlexibleBeamFem : public NeedleProperties
{

public:
    FlexibleBeamFem(uint elements);

    // Get elements number 
    uint get_elements_number(void) {return m_elements; }

    // Mass matrix getter
    arma::dmat get_mass_matrix(void){ return m_mass; }    arma::dmat mass_matrix_calculation(); 
    
    // Stiffness matrix getter
    arma::dmat get_stiffness_matrix(void){ return m_stiffness; }

    // Coriolis vector getter
    arma::dvec get_coriolis_vector(void){ return m_fvf; }

    // External force vector getter
    arma::dvec get_external_force_vector(void){ return m_qfj; }

    // Update flexible body matrices and coriolis vector
    void update(double t, arma::dvec q, arma::dvec q_dot);

    // Get element deflection
       arma::dvec get_element_deflection(double ksi, arma::dvec ej,
        uint element_id);

private:
    // Number of elements
    uint m_elements;

    // Number of dofs
    uint m_dofs;

    // Element lenght (m)
    double m_element_length;

    // Element radius (m)
    double m_element_radius;

    // Element young modulus (N / m^2)
    double m_element_young_modulus;

    // Element density (kg / m^2)
    double m_element_density;

private:
    // Pointer to to beam element object
    std::vector<BeamElement*> m_beam_element;

    // Flexible body mass matrix
    arma::dmat m_mass;
    
    // Flexible body stiffness matrix
    arma::dmat m_stiffness;

    // Flexible body coriolis-centrifugal vector
    arma::dvec m_fvf;

    // Flexible body external force vector
    arma::dvec m_qfj;

private:
    // Mass matrix of flexible body calculation 
    arma::dmat mass_matrix_calculation(arma::dvec q, arma::dvec q_dot); 

    // External force per element (bj) body frame
    arma::dvec external_force(double t, arma::dvec q, arma::dvec q_dot, uint 
        element_id);

    // External traction per element body frame
    arma::dvec external_traction(double t, arma::dvec q, arma::dvec q_dot, uint 
        element_id);
};
