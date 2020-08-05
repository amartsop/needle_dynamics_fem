#pragma once 

#include <iostream>
#include <armadillo>
#include <vector>

#include "euler_rotations.h"
#include "gnuplot-iostream.h"

template <class T, class M>
class NeedleAnimation 
{

public:
    NeedleAnimation(T *needle, M *post_processing);

    void animate(arma::dvec roa_g_g, arma::dvec euler_angles, 
        arma::dvec elastic_coordinates);

private:

    // Graphs handle
    Gnuplot m_gp;
    
    // Needle handle
    T *m_needle;

    // // Post Processing handle 
    M *m_post_processing;
    
    // Create handle geometry
    void create_handle(arma::dvec center, arma::dvec orientation);
    std::vector<arma::dmat> cylinder(arma::dvec center, arma::dvec orientation);
    std::vector<std::vector<std::string>> polygons_to_strings(std::vector<arma::dmat> polygons);
    void polygon_plot(std::vector<std::vector<std::string>> str_polygon);
};


template <class T, class M>
NeedleAnimation<T, M>::NeedleAnimation(T *needle, M *post_processing)
{
    m_gp << "set style line 1 lc rgb 'black' lw 1.5\n";
    m_gp << "set xrange [-0.4:0.4]\nset yrange [-0.4:0.4]\nset zrange [-0.4:0.4]\n";
    m_gp << "set hidden3d\n";
    m_gp << "set xlabel 'x'\nset ylabel 'y'\nset zlabel 'z'\n";
    m_gp << "set view 90, 0, 1, 1\n";
    m_gp << "set xyplane at 0\n";
    // m_gp << "set terminal gif animate delay 4\n";
    // m_gp << "set output 'test.gif'\n";

    // Needle handle
    m_needle = needle;

    // Post processing handle
    m_post_processing = post_processing;
}

template <class T, class M>
void NeedleAnimation<T, M>::animate(arma::dvec roa_g_g, arma::dvec euler_angles, 
        arma::dvec elastic_coordinates)
{
    // Caclulate the center of the rigid body
    arma::dmat rot_f_g = EulerRotations::rotation(euler_angles);
    double lx = m_needle->get_needle_offset_x();
    double lz = m_needle->get_needle_offset_z();
    arma::dvec rca_f_f = {lx, 0, -lz};
    arma::dvec roc_g_g = roa_g_g - rot_f_g * rca_f_f;
    create_handle(roc_g_g, euler_angles);

    auto beam_points_inertial = m_post_processing->get_beam_coordinates_inertial(
            roa_g_g, euler_angles, elastic_coordinates);
    
    // auto xy = beam_points.at(1);
    m_gp << "splot '-' with lines ls 1\n";
    m_gp.send1d(beam_points_inertial);
}

template <class T, class M>
void NeedleAnimation<T, M>::create_handle(arma::dvec roa_f_f, arma::dvec euler_angles)
{
    auto cylinder_pol = cylinder(roa_f_f, euler_angles);
    auto str_polygons = polygons_to_strings(cylinder_pol);
    
    int poly_num = 0;

    for (auto str_polyg_i: str_polygons)
    {
        std::string gp_input = "set object " +  std::to_string(poly_num + 1) + 
            " polygon from ";

        int count = 0; 

        for (auto coord: str_polyg_i)
        {
            if (count == str_polyg_i.size() - 1) 
            {
                gp_input += (coord);
            }
            else
            {
                gp_input += (coord + " to ");
            }

            count++; 
        }
        
        gp_input += " fillstyle transparent solid 1.0\n";
        m_gp << gp_input; 
        poly_num++;
    }
}

template <class T, class M>
std::vector<arma::dmat> NeedleAnimation<T, M>::cylinder(arma::dvec center,
    arma::dvec orientation)
{ 
    
    // Polygons vector
    std::vector<arma::dmat> polygons;

    // Needle characteristics  
    auto m_side_count = m_needle->get_handle_side_count();
    auto m_height = m_needle->get_handle_length();
    auto m_radius = m_needle->get_handle_radius();

    arma::dmat vertices = arma::zeros<arma::dmat>(2 * m_side_count, 3);


    // Cylinder vertices
    for (int i = 0; i < m_side_count; i++)
    {
        double theta = 2 * (M_PI / m_side_count) * i;
        arma::dvec vert_row1 = {-m_height / 2, m_radius * cos(theta), 
            m_radius * sin(theta)}; 
        arma::dvec vert_row2 = {m_height / 2, m_radius * cos(theta), 
            m_radius * sin(theta)}; 

        // Transform basis 
        arma::dmat rot = EulerRotations().rotation(orientation);
        vert_row1 = center + rot * vert_row1;
        vert_row2 = center + rot * vert_row2;

        vertices.row(i) = vert_row1.t();
        vertices.row(m_side_count + i) = vert_row2.t();
    }

    // Cylinder side faces
    arma::imat side_faces = arma::zeros<arma::imat>(m_side_count, 4);
    for (int i = 0; i < m_side_count - 1; i++)
    {
        arma::ivec side_faces_row = {i + 1, i + 2, m_side_count + i + 2, 
            m_side_count + i + 1}; 
        side_faces.row(i) = side_faces_row.t();
    }
    arma::ivec side_faces_last_row = {m_side_count, 1, m_side_count + 1, 2*m_side_count};
    side_faces.row(m_side_count - 1) = side_faces_last_row.t();

    // Cylinder Bottom faces        
    arma::ivec lower_bottom_face = arma::regspace<arma::ivec>(1, m_side_count);
    arma::ivec upper_bottom_face = arma::regspace<arma::ivec>(m_side_count + 1,  
        2 * m_side_count);

    arma::imat bottom_faces = arma::zeros<arma::imat>(2, m_side_count);
    bottom_faces.row(0) = lower_bottom_face.t();
    bottom_faces.row(1) = upper_bottom_face.t();

    // Vertex processing for plotting 
    for (int i = 0; i < side_faces.n_rows; i++) 
    {
        arma::ivec pol_indices = side_faces.row(i).t();
        arma::dmat polygon_coords = arma::zeros<arma::dmat>(pol_indices.n_rows, 3);

        for (int k = 0; k < pol_indices.n_rows; k++) 
        {
            int index = pol_indices(k);
            polygon_coords.row(k) = vertices.row(index - 1); 
        }
        polygons.push_back(polygon_coords);
    }

    for (int i = 0; i < bottom_faces.n_rows; i++) 
    {
        arma::ivec pol_indices = bottom_faces.row(i).t();
        arma::dmat polygon_coords = arma::zeros<arma::dmat>(pol_indices.n_rows, 3);

        for (int k = 0; k < pol_indices.n_rows; k++) 
        {
            int index = pol_indices(k);
            polygon_coords.row(k) = vertices.row(index - 1); 
        }
        polygons.push_back(polygon_coords);
    }

    return polygons;
}


template <class T, class M>
std::vector<std::vector<std::string>> NeedleAnimation<T, M>::polygons_to_strings(
    std::vector<arma::dmat> polygons)
{
    std::vector<std::vector<std::string>> str_polygons;

    for (arma::dmat polyg_coords: polygons)
    {
        std::vector<std::string> coords_vec;

        for (int j = 0; j < polyg_coords.n_rows; j++)
        {
            std::string coords_j = std::to_string(polyg_coords(j, 0)) + "," +
                std::to_string(polyg_coords(j, 1)) + "," + 
                std::to_string(polyg_coords(j, 2));

            coords_vec.push_back(coords_j);
        }

        std::string coords_n = std::to_string(polyg_coords(0, 0)) + "," +
            std::to_string(polyg_coords(0, 1)) + "," + 
            std::to_string(polyg_coords(0, 2));
        coords_vec.push_back(coords_n);
       
        str_polygons.push_back(coords_vec);
    }

    return str_polygons;
}