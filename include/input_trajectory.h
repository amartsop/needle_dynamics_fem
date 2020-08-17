#pragma once 

#include <iostream>
#include <armadillo>

class InputTrajectory
{
public:
    InputTrajectory(void);

    // Update input coordinates
    void update(double t);

    // Get linear, position, acceleration 
    arma::dvec get_linear_displacement(void) { return m_roc; }
    arma::dvec get_linear_velocity(void){ return m_roc_dot; }
    arma::dvec get_linear_acceleration(void) {  return m_roc_ddot; }

    // Get rotational position, acceleration 
    arma::dvec get_rotational_displacement(void) { return m_theta; }
    arma::dvec get_rotational_velocity(void){ return m_theta_dot; }
    arma::dvec get_rotational_acceleration(void) {  return m_theta_ddot; }

private:

    // Rigid body trajectory
    arma::dvec m_roc = {0.0, 0.0, 0.0};
    arma::dvec m_roc_dot = {0.0, 0.0, 0.0};
    arma::dvec m_roc_ddot = {0.0, 0.0, 0.0};
    
    arma::dvec m_theta = {0.0, 0.0, 0.0};
    arma::dvec m_theta_dot = {0.0, 0.0, 0.0};
    arma::dvec m_theta_ddot = {0.0, 0.0, 0.0};

private:
    // Rigid body trajectory 
    void rigid_body_trajectory(double t);
};
