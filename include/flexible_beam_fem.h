#pragma once

#include <iostream>

#include <iostream>
#include <armadillo>
#include <vector>

#include "needle_properties.hpp"
#include "dynamics_math.h"
#include "beam_element.h"

class FlexibleBeamFem : public NeedleProperties
{

public:
    FlexibleBeamFem(uint elements);

    ~FlexibleBeamFem();

    // Update flexible body matrices and coriolis vector
    void update(double t, arma::dvec q, arma::dvec q_dot);

    // Get elements number 
    uint get_elements_number(void) {return m_elements; }

    // Mass matrix getter
    arma::dmat get_mass_matrix(void){ return m_mass; }
    
    // Stiffness matrix getter
    arma::dmat get_stiffness_matrix(void){ return m_stiffness; }

    // Coriolis vector getter
    arma::dvec get_coriolis_vector(void){ return m_fvf; }

    // External force vector getter
    arma::dvec get_external_force_vector(void){ return m_qforce; }

    // Get total dofs
    uint get_total_dofs(void) { return m_dofs; }

    // Get elastic dofs 
    uint get_elastic_dofs(void) { return m_elastic_dofs; }

    // Get element deflection
       arma::dvec get_element_deflection(double xj, arma::dvec elastic_coordinates,
        uint element_id);

private:
    // Number of elements
    uint m_elements;

    // Number of rigid dofs
    const uint m_rigid_dofs = 6;

    // Number of total elastic dofs
    uint m_total_elastic_dofs;

    // Number of elastic dofs (after boundary conditios)
    uint m_elastic_dofs;

    // Number of dofs
    uint m_dofs;

    // Number of boundaries
    const uint m_boundaries = 5;

    // Boundary conditions vector 
    const arma::ivec m_lb_vec = arma::regspace<arma::ivec>(1, m_boundaries);

private:
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
    arma::dvec m_qforce;

private:
    // External force body frame (position l) FB(t, q, q_dot)
    arma::dvec external_force(uint element_id);

    // Distributed load body frame p(t, x, q , q_dot)
    arma::dvec distributed_load(double xj, uint element_id);

private:
    // State update 
    void state_update(arma::dvec q, arma::dvec q_dot);

private:
    // State & state dot
    arma::dvec m_q, m_q_dot;

    // Current time 
    double m_time;

public:
    arma::dmat get_mf31_matrix(void){ return m_mf31; }
    arma::dmat get_mf32_matrix(void){ return m_mf32; }
    arma::dmat get_mf33_matrix(void){ return m_mf33; }
    arma::dmat get_kf33_matrix(void){ return m_kf33; }
    arma::dmat get_fvf3_vector(void){ return m_fvf3; }
    arma::dmat get_qf3_vector(void){ return m_qf3; }

private:
    // Elastic components
    // Mass matrix components 
    arma::dmat m_mf31, m_mf32, m_mf33;

    // Stiffness matrix component
    arma::dmat  m_kf33;

    // Coriolis vector component 
    arma::dvec m_fvf3; 

    // External forces component 
    arma::dvec m_qf3; 

    // Elastic mass matrix calculation (constant)
    void elastic_mass_matrix_calculation(void);

    // Elastic stiffness matrix calculation
    void elastic_stiffness_matrix_calculation(void);

};
