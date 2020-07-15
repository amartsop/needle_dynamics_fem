#include "euler_rotations.h"

// Basic rotation matrix wrt x axis
arma::dmat EulerRotations::basic_rotation_x(double x)
{
    return { {1.0f, 0.0f, 0.0f}, 
        {0.0f, cos(x), -sin(x)}, 
        {0.0f, sin(x), cos(x)}};
}

// Basic rotation matrix wrt y axis
arma::dmat EulerRotations::basic_rotation_y(double x)
{
    return {{cos(x), 0.0f, sin(x)}, 
        {0.0f, 1.0f, 0.0f}, 
        {-sin(x), 0.0f, cos(x)}};
}

// Basic rotation matrix wrt z axis
arma::dmat EulerRotations::basic_rotation_z(double x)
{
    return {{cos(x), -sin(x), 0.0f}, 
        {sin(x), cos(x), 0.0f}, 
        {0.0f, 0.0f, 1.0f}};
}

// Euler rotation matrix z-y'-x''
arma::dmat EulerRotations::rotation(double phi, double theta, double psi)
{

    arma::dmat rotx =  basic_rotation_x(phi);
    arma::dmat roty = basic_rotation_y(theta);
    arma::dmat rotz = basic_rotation_z(psi);

    return (rotz * roty * rotx);
}

arma::dmat EulerRotations::rotation(arma::dvec euler_angles)
{
    return rotation(euler_angles(0), euler_angles(1), euler_angles(2));
}

// Connection of anglular velocity with euler angles derivative (w = G * theta_dot)
arma::dmat EulerRotations::G(double phi, double theta, double psi)
{
    return {{1.0, 0.0, -sin(theta)}, 
        {0.0, cos(phi), cos(theta) * sin(phi)}, 
        {0.0, -sin(phi), cos(phi) * cos(theta)}};
}

arma::dmat EulerRotations::G(arma::dvec euler_angles)
{
    return G(euler_angles(0), euler_angles(1), euler_angles(2));
}


arma::dmat EulerRotations::G_dot(arma::dvec euler_angles, arma::dvec 
    euler_angles_dot)
{
    double phi = arma::as_scalar(euler_angles(0));
    double theta = arma::as_scalar(euler_angles(1));

    double phi_dot = arma::as_scalar(euler_angles_dot(0));
    double theta_dot = arma::as_scalar(euler_angles_dot(1));

    return {{0.0, 0.0, - cos(theta) * theta_dot},
        {0.0, - sin(phi) * phi_dot, cos(theta) * cos(phi) * phi_dot - 
        sin(theta) * sin(phi) * theta_dot},
        {0.0, - cos(phi) * phi_dot, -sin(phi) * cos(theta) * phi_dot - 
        cos(phi) * sin(theta) * theta_dot}};
}