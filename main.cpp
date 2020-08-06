#include <iostream>
#include <armadillo>
#include <vector>
#include <chrono>
#include <thread>

#include "rigid_body.h"
#include "flexible_beam_fem.h"
#include "input_trajectory.h"
#include "system_fem.h"
#include "numerical_integration.hpp"

#include "include/gnuplot-iostream.h"
#include "post_processing_fem.hpp"
#include "needle_animation.hpp"


int main(int argc, char *argv[])
{
    // Number of elements 
    uint ne = 2;

    // Dofs per element 
    uint element_dofs = 10;

    // Rigid body 
    RigidBody handle;

    // Flexible body
    FlexibleBeamFem needle(ne);

    // Input trajectory
    InputTrajectory input_traj(element_dofs / 2);

    // Total system fem 
    SystemFem system(&handle, &needle, &input_traj);

    // /********************* Simulation ************************/ 
    // Initial conditions
    arma::dvec qa0 = arma::zeros<arma::dvec>(system.get_qa_size());
    arma::dvec qa0_dot = arma::zeros<arma::dvec>(system.get_qa_size());

    // State vector initialization
    std::vector<arma::dvec> state_vector;
    arma::dvec state0 = arma::join_vert(qa0, qa0_dot);
    state_vector.push_back(state0);

    system.calculate(state0, 0.0);

    // Reaction forces 
    std::vector<double> fx, fy, fz;

    // Timing
    double t_final = 10.0; // Final time (s)
    double fs = 5e3;  // Simulation frequency (Hz)
    double h = 1.0 / fs; // Integration time step (s)
    double t = 0; // Initial time (s) 
    
    // Time vector initialization 
    std::vector<double> time_vector;
    time_vector.push_back(t);

    // Problem solver 
    NumericalIntegration ni(&system, h, 4);

    // Post processing 
    PostProcessingFem post_processing_fem(&needle);

    // Iteration counter 
    uint counter = 0;

    while (t < t_final)
    {
        // System solution 
        arma::dvec x = ni.solve(state_vector.at(counter), t);
        state_vector.push_back(x);

        // Reaction forces 
        arma::dmat reaction_forces = system.get_reaction_forces();
        fx.push_back(reaction_forces(0)); fy.push_back(reaction_forces(1));
        fz.push_back(reaction_forces(2));

        // reaction_forces.push_back(system.get_reaction_forces());
        if (!x.is_finite()) {
            std::cout << "Error" << std::endl; 
            break; 
        };

        // Update time and counter
        t += h; counter += 1;

        // Update time vector 
        time_vector.push_back(t);

        std::cout << t << std::endl;
    }

    // // Plot reaction forces 
    // Gnuplot gp;
    // arma::dvec t_vec(time_vector);
    // arma::dvec fx_vec(fx);
    // arma::dvec fy_vec(fy);
    // arma::dvec fz_vec(fz);

    // arma::dmat t_fx = arma::join_horiz(t_vec.rows(0, t_vec.n_rows - 2), fx_vec);
    // arma::dmat t_fy = arma::join_horiz(t_vec.rows(0, t_vec.n_rows - 2), fy_vec);
    // arma::dmat t_fz = arma::join_horiz(t_vec.rows(0, t_vec.n_rows - 2), fz_vec);

    // gp << "plot '-' with lines \n";
    // gp.send1d(t_fz);


    // Animation
    NeedleAnimation needle_animation(&needle, &post_processing_fem);
    double animation_frequency = 30; // (Hz) Frames per sec
    double animation_period_sec = 1 / animation_frequency; // (s)
    uint animation_period =  1000 * animation_period_sec; // (ms)
    uint steps = fs / animation_frequency;

    // Clock
    auto start = std::chrono::steady_clock::now();
    double real_time = 0;

    for(size_t i = 0; i <= time_vector.size(); i = i + steps)
    {
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;

        // Current state and time 
        arma::dvec x_current = state_vector.at(i);
        double t_current = time_vector.at(i);

        // Known coordinates 
        input_traj.update(t_current);
        arma::dvec qg = input_traj.get_displacement_qg();

        // Rigid body position and orientation
        arma::dvec roa_g_g = qg.rows(0, 2);
        arma::dvec euler_angles = qg.rows(3, 5);

        // First node coordinates 
        arma::dvec first_node_pos = qg.rows(6, qg.n_rows - 1);

        // Elastic coordinates
        arma::dvec qa = x_current.rows(0, system.get_qa_size() - 1);
        arma::dvec qf = arma::join_vert(first_node_pos, qa);

        // Animation
        needle_animation.animate(roa_g_g, euler_angles, qf);

        // Delay
        std::this_thread::sleep_for(std::chrono::milliseconds(animation_period));

        // Calculate Real time
        double real_time = real_time + elapsed_seconds.count();

        // Update time
        start = end;
    }
    
}