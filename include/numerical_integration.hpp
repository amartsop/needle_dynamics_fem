#pragma once 

#include <iostream>
#include <vector>
#include <armadillo>


template <class T>
class NumericalIntegration
{
public:

    NumericalIntegration(T *model, double integration_step, uint state_vec_size);

    /************************** Explicit Methods *************************/
    // Runge-Kutta family
    arma::dvec explicit_euler(double t, arma::dvec state);
    arma::dvec explicit_midpoint(double t, arma::dvec state);
    arma::dvec explicit_trapezoid(double t, arma::dvec state);
    arma::dvec runge_kutta_3(double t, arma::dvec state);
    arma::dvec runge_kutta_4(double t, arma::dvec state);
    arma::dvec runge_kutta_5(double t, arma::dvec state);

    // Adams-Bashforth family
    arma::dvec adams_bashforth_2(double t, arma::dvec state);
    arma::dvec adams_bashforth_4(double t, arma::dvec state);
    
    // Predictor-Correctors
    arma::dvec predictor_corrector_2(double t, arma::dvec state);
    arma::dvec predictor_corrector_4(double t, arma::dvec state);

    /************************** Implicit Methods *************************/
    arma::dvec implicit_euler(double t, arma::dvec state);
    arma::dvec implicit_midpoint(double t, arma::dvec state);
    arma::dvec implicit_trapezoid(double t, arma::dvec state);
    arma::dvec implicit_runge_kutta_2(double t, arma::dvec state);
    arma::dvec implicit_runge_kutta_4(double t, arma::dvec state);

private:

    /************************** Butcher tableau *************************/

    //************************* Explicit Methods ****************************//
    // Explicit (Forward) Euler
    const arma::dmat e_euler_a = {0.0};
    const arma::dvec e_euler_b = {1.0};
    const arma::dvec e_euler_c = {0.0};

    // Explicit midpoint method
    const arma::dmat e_midpoint_a = {{0.0, 0.0}, {1.0 / 2.0, 0.0}};
    const arma::dvec e_midpoint_b = {0.0, 1.0};
    const arma::dvec e_midpoint_c = {0.0, 1.0 / 2.0};

    // Explicit trapezoid method
    const arma::dmat e_trapezoid_a = {{0.0, 0.0}, {1.0, 0.0}};
    const arma::dvec e_trapezoid_b = {1.0 / 2.0, 1.0 / 2.0};
    const arma::dvec e_trapezoid_c = {0.0, 1.0};

    // Explicit third order Runge-Kutta
    const arma::dmat e_runge3_a = {{0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0}, {1.0 / 4.0, 1.0 / 4.0, 0.0}};
    const arma::dvec e_runge3_b = {1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0};
    const arma::dvec e_runge3_c = {0.0, 1.0, 1.0 / 2.0};

    // Explicit fourth order Runge-Kutta
    const arma::dmat e_runge4_a = {{0.0, 0.0, 0.0, 0.0},
        {1.0 / 2.0, 0.0, 0.0, 0.0}, {0.0, 1.0 / 2.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0}};
    const arma::dvec e_runge4_b = {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
    const arma::dvec e_runge4_c = {0.0, 1.0 / 2.0, 1.0 / 2.0, 1.0};

    // Explicit fifth order Runge-Kutta (Fehlberg)
    const arma::dmat e_runge5_a = {{0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0 / 4.0, 0.0, 0.0, 0.0, 0.0}, {3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0},
        {1932.0 / 2197.0 , - 7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0},
        {439.0 / 216.0, - 8.0, 3680.0 / 513.0, - 845.0 / 4104.0, 0.0},
        {- 8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0} };
    const arma::dvec e_runge5_b = {16.0 / 135.0, 0.0, 6656.0 / 12825.0,
        28561.0 / 56430.0, - 9.0 / 50.0, 2.0 / 55.0};
    const arma::dvec e_runge5_c = {0.0, 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0,
        1.0, 1.0 / 2.0};

    arma::dvec generic_explicit_runge_kutta(double t, arma::dvec state,
        arma::dmat coef_a, arma::dvec coef_b, arma::dvec coef_c);

    /************************** Implicit Methods *************************/
    // Implicit (Backward) Euler
    const arma::dmat i_euler_a = {1.0};
    const arma::dvec i_euler_b = {1.0};
    const arma::dvec i_euler_c = {1.0};

    // Implicit midpoint
    const arma::dmat i_midpoint_a = {1.0};
    const arma::dvec i_midpoint_b = {1.0};
    const arma::dvec i_midpoint_c = {1.0};

    // Implicit trapezoid method
    const arma::dmat i_trapezoid_a = {{0.0, 0.0}, {1.0 / 2.0, 1.0 / 2.0}};
    const arma::dvec i_trapezoid_b = {1.0 / 2.0, 1.0 / 2.0};
    const arma::dvec i_trapezoid_c = {0.0, 1.0};

    // Implicit second order Runge-Kutta
    const arma::dmat i_runge2_a = {{1.0 / 2.0, 0.0}, {-1.0 / 2.0, 2.0}};
    const arma::dvec i_runge2_b = {- 1.0 / 2.0, 3.0 / 2.0};
    const arma::dvec i_runge2_c = {1.0 / 2.0, 3.0 / 2.0};

    // Implicit fourth order Runge-Kutta
    const arma::dmat i_runge4_a = {{1.0 / 2.0, 0.0, 0.0, 0.0},
        {1.0 / 6.0, 1.0 / 2.0, 0.0, 0.0},
        {- 1.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0, 0.0},
        {3.0 / 2.0, - 3.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0}};
    const arma::dvec i_runge4_b = {3.0 / 2.0, - 3.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0};
    const arma::dvec i_runge4_c = {3.0 / 2.0, - 3.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0};

    arma::dvec generic_implicit_runge_kutta(double t, arma::dvec state,
        arma::dmat coef_a, arma::dvec coef_b, arma::dvec coef_c);

private:

    // Model handle
    T *m_model;

    // Integration step 
    double m_integration_step;

    // State vector 
    arma::dvec m_y;

    // Iterations counter 
    uint64_t m_iterations;

    // Previous evaluation holder 
    arma::dvec m_previous_evaluations[10];

    // Tolerance of Newton-Raphson method
    double m_tol = 10e-5;
};

template <class T>
NumericalIntegration<T>::NumericalIntegration(T *model, double integration_step,
    uint state_vec_size)
{
    // Model to be integrated  
    m_model = model;
    
    // Integration step 
    m_integration_step = integration_step;

    // Iterations 
    m_iterations = 0;
}


//************************* Runge-Kutta Methods ****************************//

// Euler method
template <class T>
arma::dvec NumericalIntegration<T>::explicit_euler(double t, arma::dvec state)
{   
    return generic_explicit_runge_kutta(t, state, e_euler_a, e_euler_b,
        e_euler_c);
}

// Explicit midpoint method
template <class T>
arma::dvec NumericalIntegration<T>::explicit_midpoint(double t, arma::dvec state)
{   
    return generic_explicit_runge_kutta(t, state, e_midpoint_a, e_midpoint_b,
        e_midpoint_c);
}

// Explicit trapezoid method
template <class T>
arma::dvec NumericalIntegration<T>::explicit_trapezoid(double t, arma::dvec state)
{   
    return generic_explicit_runge_kutta(t, state, e_trapezoid_a, e_trapezoid_b,
        e_trapezoid_c);
}

// Third order Runge-Kutta
template <class T>
arma::dvec NumericalIntegration<T>::runge_kutta_3(double t, arma::dvec state)
{   
    return generic_explicit_runge_kutta(t, state, e_runge3_a, e_runge3_b,
        e_runge3_c);
}

// Fourth order Runge-Kutta
template <class T>
arma::dvec NumericalIntegration<T>::runge_kutta_4(double t, arma::dvec state)
{   
    return generic_explicit_runge_kutta(t, state, e_runge4_a, e_runge4_b,
        e_runge4_c);
}

// Fifth order Runge-Kutta
template <class T>
arma::dvec NumericalIntegration<T>::runge_kutta_5(double t, arma::dvec state)
{   
    return generic_explicit_runge_kutta(t, state, e_runge5_a, e_runge5_b,
        e_runge5_c);
}

// Generic explicit Runge-Kutta
template <class T>
arma::dvec NumericalIntegration<T>::generic_explicit_runge_kutta(double t,
    arma::dvec state, arma::dmat coef_a, arma::dvec coef_b, arma::dvec coef_c)
{
    arma::dvec biki_sum = arma::zeros(state.n_rows);
    uint s = coef_b.n_rows; arma::dvec ki_vec[s];

    ki_vec[0] = m_model->f(t, state);

    for(uint i = 1; i <= s; i++)
    {
        arma::dvec aijkj_sum = arma::zeros(state.n_rows);

        for (uint j = 1; j <= i - 1; j++)
        {
            aijkj_sum += coef_a(i - 1, j - 1) * ki_vec[j - 1];
        }
        ki_vec[i - 1] = m_model->f(t + coef_c(i - 1) * m_integration_step, 
            state + m_integration_step * aijkj_sum);

        biki_sum += coef_b(i - 1) * ki_vec[i - 1];
    }

    return (state + m_integration_step * biki_sum);
}

//************************* Implicit Methods ****************************//

// Implicit Euler method
template <class T>
arma::dvec NumericalIntegration<T>::implicit_euler(double t, arma::dvec state)
{   
    return generic_implicit_runge_kutta(t, state, i_euler_a, i_euler_b,
        i_euler_c);
}


// Implicit midpoint method
template <class T>
arma::dvec NumericalIntegration<T>::implicit_midpoint(double t, arma::dvec state)
{   
    return generic_implicit_runge_kutta(t, state, i_midpoint_a, i_midpoint_b,
        i_midpoint_c);
}


// Implicit trapezoid method
template <class T>
arma::dvec NumericalIntegration<T>::implicit_trapezoid(double t, arma::dvec state)
{   
    return generic_implicit_runge_kutta(t, state, i_trapezoid_a, i_trapezoid_b,
        i_trapezoid_c);
}

// Implicit third order Runge-Kutta
template <class T>
arma::dvec NumericalIntegration<T>::implicit_runge_kutta_2(double t, arma::dvec state)
{   
    return generic_implicit_runge_kutta(t, state, i_runge2_a, i_runge2_b,
        i_runge2_c);
}

// Implicit fourth order Runge-Kutta
template <class T>
arma::dvec NumericalIntegration<T>::implicit_runge_kutta_4(double t, arma::dvec state)
{   
    return generic_implicit_runge_kutta(t, state, i_runge4_a, i_runge4_b,
        i_runge4_c);
}

// Generic (diagonal) implicit Runge-Kutta
template <class T>
arma::dvec NumericalIntegration<T>::generic_implicit_runge_kutta(double t,
    arma::dvec state, arma::dmat coef_a, arma::dvec coef_b, arma::dvec coef_c)
{
    arma::dvec biki_sum = arma::zeros(state.n_rows);
    uint s = coef_b.n_rows; arma::dvec ki_vec[s];

    for(uint i = 1; i <= s; i++)
    {
        arma::dvec aijkj_sum = arma::zeros(state.n_rows);

        for (uint j = 1; j <= i - 1; j++)
        {
            aijkj_sum += coef_a(i - 1, j - 1) * ki_vec[j - 1];
        }

        // Initial state guess
        arma::dvec ki = state;

        // Next time step 
        double t_predict = t + coef_c(i - 1) * m_integration_step;

        // R function evaluation 
        arma::dvec rki = state + m_integration_step * (aijkj_sum + 
            coef_a(i - 1, i - 1) * ki);

        // Function evaluation
        arma::dvec g = ki - m_model->f(t_predict, rki);

        // Residual 
        double res = arma::norm(g);

        while (res > m_tol)
        {
            // Jacobian
            arma::dmat jac = m_model->dfdx(t_predict, rki);

            arma::dmat alpha = arma::eye(jac.n_rows, jac.n_cols) -
                m_integration_step * coef_a(i - 1, i - 1) * jac;

            // Update solution
            ki = ki - arma::solve(alpha, g);

            // Update function
            rki = state + m_integration_step * (aijkj_sum + 
                coef_a(i - 1, i - 1) * ki);
            g = ki - m_model->f(t_predict, rki);

            // Update residual
            res = arma::norm(g);
        }

        ki_vec[i - 1] = ki;

        biki_sum += coef_b(i - 1) * ki_vec[i - 1];
    }

    return (state + m_integration_step * biki_sum);
}




























//************************* Predictor-Correctors ****************************//
template <class T>
arma::dvec NumericalIntegration<T>::predictor_corrector_2(double t,
    arma::dvec state)
{
    // Predictor
    arma::dvec x_predict = adams_bashforth_2(t, state);
    double t_predict = t + m_integration_step;
    arma::dvec f_predicted = m_model->f(t_predict, x_predict);

    // Current 
    arma::dvec f_current = m_model->f(t, state);
    
    // Corrector
    return (state + (m_integration_step / 2.0) * (f_current + f_predicted));
}
 
template <class T>
arma::dvec NumericalIntegration<T>::predictor_corrector_4(double t, 
    arma::dvec state)
{
    arma::dvec x_predict = adams_bashforth_4(t, state);
    double t_predict = t + m_integration_step;
    arma::dvec f_predicted = m_model->f(t_predict, x_predict);

    //Current 
    arma::dvec f_current = m_model->f(t, state);

    if(m_iterations < 3) 
    {
        return x_predict;
    }
    else
    {
        // Previous
        arma::dvec f_prev_2 = m_previous_evaluations[0];
        arma::dvec f_prev_1 = m_previous_evaluations[1];
    
        // Corrector
        return (state + (m_integration_step / 24.0) * (9.0 * f_predicted +
            19.0 * f_current  - 5.0 * f_prev_1 + f_prev_2));
    }
    
}

//************************* Adams-Bushforth Methods ****************************//
template <class T>
arma::dvec NumericalIntegration<T>::adams_bashforth_2(double t, arma::dvec state)
{
    arma::dvec output;

    if (m_iterations == 0)
    {
        m_previous_evaluations[0] = m_model->f(t, state);


        // Calculation of state 1
        output = explicit_trapezoid(t, state);
    }
    else
    {
        arma::dvec f_prev = m_previous_evaluations[0];
        arma::dvec f_current = m_model->f(t, state);
        m_previous_evaluations[0] = f_current;

        output = state + (m_integration_step / 2.0) * (3 * f_current - f_prev);
    }

    // Update iterations
    m_iterations += 1;

    return output;
}

template <class T>
arma::dvec NumericalIntegration<T>::adams_bashforth_4(double t, arma::dvec state)
{
    arma::dvec output;

    if (m_iterations < 3)
    {
        m_previous_evaluations[m_iterations] = m_model->f(t, state);

        // Calculation of state 1
        output = runge_kutta_4(t, state);
    }
    else
    {
        arma::dvec f_prev_3 = m_previous_evaluations[0];
        arma::dvec f_prev_2 = m_previous_evaluations[1];
        arma::dvec f_prev_1 = m_previous_evaluations[2];
        arma::dvec f_current = m_model->f(t, state);
        
        m_previous_evaluations[0] = f_prev_2;
        m_previous_evaluations[1] = f_prev_1;
        m_previous_evaluations[2] = f_current;

        output = state + (m_integration_step / 24.0) * (55.0 * f_current - 
            59.0 * f_prev_1 + 37.0 * f_prev_2 - 9.0 * f_prev_3);
    }

    // Update iterations
    m_iterations += 1;

    return output;
}