#pragma once 

#include <iostream>
#include <armadillo>

class EulerRotations
{
public:
    

    // Basic rotation matrix wrt x axis
    static arma::dmat basic_rotation_x(double x);
    
    // Basic rotation matrix wrt y axis
    static arma::dmat basic_rotation_y(double x);

    // Basic rotation matrix wrt z axis
    static arma::dmat basic_rotation_z(double x);

    // Euler rotation matrix z-y'-x''
    static arma::dmat rotation(double phi, double theta, double psi);
    static arma::dmat rotation(arma::dvec euler_angles);

    // Connection of anglular velocity with euler angles derivative (w = G * theta_dot)
    static arma::dmat G(double phi, double theta, double psi);
    static arma::dmat G(arma::dvec euler_angles);

    // Time derivative of G matrix
    static arma::dmat G_dot(arma::dvec euler_angles, arma::dvec euler_angles_dot);

private:

};
