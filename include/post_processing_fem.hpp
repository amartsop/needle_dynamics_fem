#pragma once 

#include <iostream>
#include <armadillo>
#include <vector>

#include "euler_rotations.h"
#include "gnuplot-iostream.h"
#include "dynamics_math.h"

template <class T>

class PostProcessingFem
{
public:

    // Postprocessing for plotting and analysis (Rayleigh-Ritz)
    PostProcessingFem(T *needle);
    
    // Get the beam coordinates in the floating frame of reference
    arma::dmat get_beam_coordinates_ffr(arma::dmat elastic_coordinates);

    // Get the beam coordinates in the inertial frame of reference
    arma::dmat get_beam_coordinates_inertial(arma::dvec roc_g_g, 
        arma::dvec euler_angles, arma::dvec elastic_coordinates) ;

private:

    // Needle handle 
    T* m_needle_ptr;

    // Number of dofs 
    uint m_dofs_num;

    // Number of elements
    uint m_elements_num;
   
    // Number of points in each element
    const int npel = 50;     

    // Beam length 
    double m_beam_lenght;

    // Element lenght
    double m_element_length;
};

template <class T>
PostProcessingFem<T>::PostProcessingFem(T *needle)
{
    // Needle handle
    m_needle_ptr = needle;

    // Elements number 
    m_elements_num = m_needle_ptr->get_elements_number();

    // Number of dofs 
    m_dofs_num = 5 * (m_elements_num + 1);

    // Beam length
    m_beam_lenght = m_needle_ptr->get_needle_length();

    // Element length
    m_element_length = m_beam_lenght / (double)m_elements_num;
}

template <class T>
arma::dmat PostProcessingFem<T>::get_beam_coordinates_ffr(arma::dmat elastic_coordinates)
{
    // Beam points 
    std::vector<double> m_beam_points_x, m_beam_points_y, m_beam_points_z;

    // Beam points armadillo 
    arma::dvec rx, ry, rz;
    
    // Discretize element
    arma::dvec x2j_vec = arma::linspace<arma::dvec> (0, m_element_length, npel);

    for (int j = 1; j <= m_elements_num; j++)
    {
        for (int i = 0; i < npel; i++) 
        {   
            // Deformation of point pj
            arma::dvec rp0pj_f_f =
                m_needle_ptr->get_element_deflection(x2j_vec(i),
                elastic_coordinates, j);

            // Initial position of pj wrt to f origin
            arma::dvec rap0j_f_f = { (double) (j - 1) * m_element_length + 
                x2j_vec(i), 0.0f, 0.0f};
            
            // Final position of pj wrt to f origin
            arma::dvec rapj_f_f = rap0j_f_f + rp0pj_f_f;
    
            // Final position of pj wrt to f origin
            m_beam_points_x.push_back(rapj_f_f(0));
            m_beam_points_y.push_back(rapj_f_f(1));
            m_beam_points_z.push_back(rapj_f_f(2));
        }
    }

    // Armadillo matrices of beam elements 
    rx = arma::vec(m_beam_points_x); ry = arma::vec(m_beam_points_y);
    rz = arma::vec(m_beam_points_z);

    return (arma::join_horiz(rx, ry, rz));
}

template <class T>
arma::dmat PostProcessingFem<T>::get_beam_coordinates_inertial(arma::dvec roa_g_g, 
    arma::dvec euler_angles, arma::dvec elastic_coordinates)
{
    // Rigid body coordinates
    arma::dmat rot_f_g = EulerRotations::rotation(euler_angles);
    
    // Beam points 
    std::vector<double> m_beam_points_x, m_beam_points_y, m_beam_points_z;

    // Beam points armadillo 
    arma::dvec rx, ry, rz;

    // Discretize element
    arma::dvec x2j_vec = arma::linspace<arma::dvec> (0, m_element_length, npel);

    for (int j = 1; j <= m_elements_num; j++)
    {
        for (int i = 0; i < npel; i++) 
        { 
            // Deformation of point pj
            arma::dvec rp0pj_f_f = 
                m_needle_ptr->get_element_deflection(x2j_vec(i),
                elastic_coordinates, j);

            // Initial position of pj wrt to f origin
            arma::dvec rap0j_f_f = { (double) (j - 1) * m_element_length + 
                x2j_vec(i), 0.0f, 0.0f};
            
            // Final position of pj wrt to f origin
            arma::dvec rapj_f_f = rap0j_f_f + rp0pj_f_f;

            // Final position of pj wrt to G origin 
            arma::dvec ropj_g_g = roa_g_g + rot_f_g *  rapj_f_f;
            
            // Final position of pj wrt to f origin
            m_beam_points_x.push_back(ropj_g_g(0));
            m_beam_points_y.push_back(ropj_g_g(1));
            m_beam_points_z.push_back(ropj_g_g(2));
        }
    }

    // Armadillo matrices of beam elements 
    rx = arma::vec(m_beam_points_x); ry = arma::vec(m_beam_points_y);
    rz = arma::vec(m_beam_points_z);

    return (arma::join_horiz(rx, ry, rz));
}
