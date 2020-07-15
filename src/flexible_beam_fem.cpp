#include "flexible_beam_fem.h"
#include "vector"

FlexibleBeamFem::FlexibleBeamFem(uint elements)
{
    // Number of elements
    m_elements = elements;

    // Element length
    m_element_length = m_needle_length / (double) m_elements;

    // Eement radius
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

    // Number of dofs 
    m_dofs = beam_element.at(0)->get_dofs_number() + 6;

    // Mass matrix initialization 
    m_mass = arma::zeros(m_dofs, m_dofs);

    // Stiffness matrix initialization 
    m_stiffness = arma::zeros(m_dofs, m_dofs);

    // Coriolis-centrifugal vector initialization 
    m_fvf = arma::zeros(m_dofs, 1);

    // External force vector initialization 
    m_qfj = arma::zeros(m_dofs, 1);
}

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
        // External forces and truction
        arma::dvec fbj_f = external_force(t, q, q_dot, i + 1);
        arma::dvec tsj_f = external_traction(t, q, q_dot, i + 1);

        // Update element j
        (m_beam_element.at(i))->update(q, q_dot, fbj_f, tsj_f);

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

arma::dvec FlexibleBeamFem::external_force(double time, arma::dvec q,
    arma::dvec q_dot, uint element_id)
{
    double fx, fy, fz;

    if (element_id == m_elements)
    {
        fx = 0.0 * time;
        fz = 0.0 * time;
        fy = 0.0 * time;
        if (fz >= 0.5) { fz = 0.0; }
        if (fy >= 0.5) { fy =  0.0; }
        if (fx >= 0.0) { fx =  0.0; }

    }
    else
    {
        fx = 0.0; fy = 0.0; fz = 0.0;
    }

    return {fx, fy, fz};
}

arma::dvec FlexibleBeamFem::external_traction(double time, arma::dvec q,
    arma::dvec q_dot, uint element_id)
{
    return {0.0, 0.0, 0.0};
}