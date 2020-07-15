#pragma once 

#include <iostream>
#include <armadillo>

class InputTrajectory
{
public:
    InputTrajectory(uint dofs_per_node);

    // Update input coordinates
    void update(double t);

    // Get known coordinates 
    arma::dvec get_displacement_qg(void) { return m_qg; }
    arma::dvec get_velocity_qg_dot(void){ return m_qg_dot; }
    arma::dvec get_acceleration_qg_ddot(void) {  return m_qg_ddot; }

private:
    // Dofs per node 
    uint m_dofs_per_node;

    // Rigid body trajectory
    arma::dvec m_roa = {0.0, 0.0, 0.0};
    arma::dvec m_roa_dot ={0.0, 0.0, 0.0};
    arma::dvec m_roa_ddot ={0.0, 0.0, 0.0};
    
    arma::dvec m_theta = {0.0, 0.0, 0.0};
    arma::dvec m_theta_dot = {0.0, 0.0, 0.0};
    arma::dvec m_theta_ddot = {0.0, 0.0, 0.0};

    // Needle first node trajectory
    arma::dvec first_node_position;
    arma::dvec first_node_velocity;
    arma::dvec first_node_acceleration;

private:
    // Known coordinates for the total system
    arma::dvec m_qg, m_qg_dot, m_qg_ddot;

    // Rigid body trajectory 
    void rigid_body_trajectory(double t);
};
