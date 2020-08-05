#include "input_trajectory.h"


InputTrajectory::InputTrajectory(uint dofs_per_node)
{
    // Dofs per node 
    m_dofs_per_node = dofs_per_node;

    // First node coordinates
    first_node_position = arma::zeros(m_dofs_per_node);
    first_node_velocity = arma::zeros(m_dofs_per_node);
    first_node_acceleration = arma::zeros(m_dofs_per_node);
}

void InputTrajectory::update(double t)
{
    rigid_body_trajectory(t);

    // Known postions
    m_qg = arma::join_vert(m_roa, m_theta, first_node_position);

    // Known velocities
    m_qg_dot = arma::join_vert(m_roa_dot, m_theta_dot, first_node_velocity);

    // Known accelerations
    m_qg_ddot = arma::join_vert(m_roa_ddot, m_theta_ddot, first_node_acceleration);

}

void InputTrajectory::rigid_body_trajectory(double t)
{
    
    // /******************* Position *******************/

    // Position amplitude (m)
   arma::dvec a_p = {0.0, 0.0, 0.2};

    // Position frequency (Hz)
    arma::dvec f_p = {1.0, 0.0, 5.0};

    // Position phase (rad)
    arma::dvec phi_p = {0.0, 0.0, 0.0};
    
    /******************* Orientation *******************/
    // Euler angles amplitude (rad)
    arma::dvec a_o = {0.0, 0.0, 0.0};

    // Euler angles frequency (Hz)
    arma::dvec f_o = {0.0, 10.0, 0.0};

    // Euler angles phase (rad)
    arma::dvec phi_o = {0.0, 0.0, 0.0};

    /******************* Trajectory functions *******************/
    
    for (int i = 0; i < 3; i++)
    {
        // Rigid body translational trajectory
        double a_dot = 2.0 * M_PI * f_p(i);
        double a = a_dot * t + phi_p(i);

        m_roa(i) = a_p(i) * sin(a);
        m_roa_dot(i) = a_p(i) * a_dot * cos(a);
        m_roa_ddot(i) = - a_p(i) * powf(a_dot, 2.0) * sin(a);

        // Rigid body rotational trajectory
        double b_dot = 2.0 * M_PI * f_o(i);
        double b = b_dot * t + phi_o(i);
    
        m_theta(i) = a_o(i) * sin(b);
        m_theta_dot(i) = a_o(i) * b_dot * cos(b);
        m_theta_ddot(i) = - a_o(i) * powf(b_dot, 2.0) * sin(b);
    }

}
