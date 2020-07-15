#pragma once

#include <iostream>
#include <armadillo>

#include "dynamics_math.h"
#include "state.h"
#include "needle_properties.hpp"


class RigidBody : public NeedleProperties 
{

public:
    RigidBody();

    // Mass matrix getter
    arma::dmat get_mass_matrix(void){ return m_mass; }
    
    // Coriolis vector getter
    arma::dvec get_coriolis_vector(void){ return m_fvr; }

    // Update rigid body matrices and coriolis vector
    void update(double t, arma::dvec q, arma::dvec q_dot);

private:
    // Distance vector of reference frame origin and centre of mass (wrt reference frame)
    arma::dvec m_r_ac_f_f;

    // Handle length 
    double m_length;

    // Handle radius 
    double m_radius;

    // Handle mass 
    double m_mr;

    // Handle inertia tensor (wrt to point A in reference frame)
    arma::dmat m_ir;

private:
    // Rigid body mass matrix
    arma::dmat m_mass;

    // Coriolis-Centrifugal vector
    arma::dvec m_fvr;

    // External forces 
    arma::dvec m_qr;

private:
    // Mass matrix calculation
    void mass_matrix_calculation(arma::dvec q, arma::dvec q_dot); 

    // Coriolis-Centrifugal calculation
    void coriolis_vector_calculation(arma::dvec q, arma::dvec q_dot);
};