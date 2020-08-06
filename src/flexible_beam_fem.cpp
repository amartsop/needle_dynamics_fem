#include "flexible_beam_fem.h"
#include "vector"

FlexibleBeamFem::FlexibleBeamFem(uint elements)
{
    // Number of elements
    m_elements = elements;

    // Element length
    m_element_length = m_needle_length / (double) m_elements;

    // Element radius
    m_element_radius = m_needle_radius;

    // Element young modulus
    m_element_young_modulus = m_needle_young_modulus;

    // Element density)
    m_element_density = m_needle_density;

    // Beam elements
    BeamElement* beam_element_ptr = nullptr;
    std::vector<BeamElement*> beam_element;
    
    for (uint i = 0; i < m_elements; i++)
    {
        beam_element_ptr = new BeamElement(i + 1, m_elements, m_element_length, 
            m_element_radius, m_element_young_modulus, m_element_density);
        beam_element.push_back(beam_element_ptr);
    }
    m_beam_element = beam_element;

    // Number of elastic dofs
    m_elastic_dofs = beam_element.at(0)->get_dofs_number();

    // Number of dofs 
    m_dofs = m_elastic_dofs + m_rigid_dofs;
    
    // Mass matrix initialization 
    m_mass = arma::zeros(m_dofs, m_dofs);

    // Stiffness matrix initialization 
    m_stiffness = arma::zeros(m_dofs, m_dofs);

    // Coriolis-centrifugal vector initialization 
    m_fvf = arma::zeros(m_dofs, 1);

    // External force vector initialization 
    m_qfj = arma::zeros(m_dofs, 1);
}


// Update beam equations
void FlexibleBeamFem::update(double t, arma::dvec q, arma::dvec q_dot)
{

    // Mass matrix initialization 
    arma::dmat mass_total = arma::zeros(m_dofs, m_dofs);

    // Stiffness matrix initialization 
    arma::dmat stiffness_total = arma::zeros(m_dofs, m_dofs);

    // Coriolis-centrifugal vector initialization 
    arma::dvec fvf_total = arma::zeros(m_dofs, 1);

    // External force vector initialization 
    arma::dvec qfj_total = arma::zeros(m_dofs, 1);


    for (uint i = 0; i < m_elements; i++)
    {
        // Update element j state, mass matrix and coriolis vector
        (m_beam_element.at(i))->update(t, q, q_dot);

        // Calculation of external forces 

        /**************** Integrals estimation ****************/
        uint grid_size = 100;
        double a = 0.0; double b = m_element_length;
        double dxj = (b - a) / (double) grid_size;

        // Grid 
        arma::dvec xj_vec = arma::linspace(a, b, grid_size);

        // Simpson's rule
        arma::dvec integral_qfj1 = arma::zeros(3, 1);
        arma::dvec integral_qfj2 = arma::zeros(3, 1);
        arma::dvec integral_qfj3 = arma::zeros(m_elastic_dofs, 1);

        for (uint k = 0; k < grid_size; k++) 
        {
            double xj = arma::as_scalar(xj_vec(k));

            arma::dvec g1 = distributed_load(t, xj, q, q_dot, i + 1);

            arma::dvec g2 = (m_beam_element.at(i))->dj_x(xj) *
                distributed_load(t, xj, q, q_dot, i + 1);

            arma::dvec g3 = (m_beam_element.at(i))->shape_function(xj).t() *
                distributed_load(t, xj, q, q_dot, i + 1);

            if (k == 0 || k == grid_size - 1) {
                integral_qfj1 += g1; integral_qfj2 += g2; integral_qfj3 += g3;
            }
            else if (k % 2 != 0)
            {
                integral_qfj1 += 4.0 * g1; integral_qfj2 += 4.0 * g2;
                integral_qfj3 += 4.0 * g3;
            }
            else
            {
                integral_qfj1 += 2.0 * g1; integral_qfj2 += 2.0 * g2;
                integral_qfj3 += 2.0 * g3;
            } 
        }

        integral_qfj1 *= (dxj / 3.0); integral_qfj2 *= (dxj / 3.0);
        integral_qfj3 *= (dxj / 3.0); 

        // External forces and truction
        arma::dvec fbj_f = external_force(t, q, q_dot, i + 1);

        // Update external forces 
        (m_beam_element.at(i))->calculate_external_forces(integral_qfj1,
            integral_qfj2, integral_qfj3, fbj_f);

        // Get mass matrix of element j
        arma::dmat mfj = (m_beam_element.at(i))->get_mass_matrix();
        
        // Get stiffness matrix of element j
        arma::dmat kfj = (m_beam_element.at(i))->get_stiffness_matrix();
    
        // Get coriolis-centrifugal vector of element j
        arma::dmat fvfj = (m_beam_element.at(i))->get_coriolis_vector();

        // Get external force vector of element j
        arma::dvec qfj = (m_beam_element.at(i))->get_external_force_vector();

        // Update mass matrix of beam 
        mass_total += mfj;

        // Update stiffness matrix of beam 
        stiffness_total += kfj;

        // Update coriolis-centrifugal vector of beam 
        fvf_total += fvfj;

        // Update external force vector of beam 
        qfj_total += qfj;
    }

    m_mass = mass_total;
    m_stiffness = stiffness_total;
    m_fvf = fvf_total;
    m_qfj = qfj_total;
}


// External force calculation
arma::dvec FlexibleBeamFem::external_force(double t, arma::dvec q,
    arma::dvec q_dot, uint element_id)
{
    double fx, fy, fz;

    if (element_id == m_elements)
    {
        fx = 0.0;
        fy = 0.0 * t;
        fz = 0.0 * t;

        if (fx >= 0.0) { fx =  0.0; }
        if (fy >= 0.5) { fy =  0.5; }
        if (fz >= 0.5) { fz = 0.5; }

    }
    else
    {
        fx = 0.0; fy = 0.0; fz = 0.0;
    }

    return {fx, fy, fz};
}


// Distributed load body frame pj(t, x, q , q_dot)
arma::dvec FlexibleBeamFem::distributed_load(double t, double xj, arma::dvec q,
    arma::dvec q_dot, uint element_id)
{

    double px, py, pz;

    px = 0.0; py = 0.0; pz = 0.0;

    return {px, py, pz};
}

arma::dvec FlexibleBeamFem::get_element_deflection(double xj, arma::dvec 
    elastic_coordinates, uint element_id)
 {
    return m_beam_element.at(element_id - 1)->get_deflection(xj,
        elastic_coordinates);
}