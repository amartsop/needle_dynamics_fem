#include "input_trajectory.h"


InputTrajectory::InputTrajectory(void)
{
}

void InputTrajectory::update(double t)
{
    rigid_body_trajectory(t);

}

void InputTrajectory::rigid_body_trajectory(double t)
{
    
    // /******************* Position *******************/

    // Position amplitude (m)
   arma::dvec a_p = {0.0, 0.0, 0.0};

    // Position frequency (Hz)
    arma::dvec f_p = {1.0, 5.0, 2.0};

    // Position phase (rad)
    arma::dvec phi_p = {0.0, 0.0, 0.0};
    
    /******************* Orientation *******************/
    // Euler angles amplitude (rad)
    arma::dvec a_o = {0.0, 0.0, 0.0};

    // Euler angles frequency (Hz)
    arma::dvec f_o = {1.0, 5.0, 1.1};

    // Euler angles phase (rad)
    arma::dvec phi_o = {0.0, 0.0, 0.0};

    /******************* Trajectory functions *******************/
    
    for (int i = 0; i < 3; i++)
    {
        // Rigid body translational trajectory
        double a_dot = 2.0 * M_PI * f_p(i);
        double a = a_dot * t + phi_p(i);

        m_roc(i) = a_p(i) * sin(a);
        m_roc_dot(i) = a_p(i) * a_dot * cos(a);
        m_roc_ddot(i) = - a_p(i) * powf(a_dot, 2.0) * sin(a);

        // Rigid body rotational trajectory
        double b_dot = 2.0 * M_PI * f_o(i);
        double b = b_dot * t + phi_o(i);
    
        m_theta(i) = a_o(i) * sin(b);
        m_theta_dot(i) = a_o(i) * b_dot * cos(b);
        m_theta_ddot(i) = - a_o(i) * powf(b_dot, 2.0) * sin(b);
    }

}
