#pragma once 

class NeedleProperties
{
public:
    // Gravity constant (m / s^2)
    const double m_grav = 9.80665 ;

public:
    // Handle geometry
    static double get_handle_radius(void) { return m_handle_radius; }
    static double get_handle_length(void) { return m_handle_length; }
    static int get_handle_side_count(void) { return m_side_count; }
    static int get_handle_mass(void) { return m_handle_mass; }

    // Needle geometry
    static double get_needle_offset_z(void) {return m_needle_lz; }
    static double get_needle_offset_x(void) {return m_needle_lx; }
    static double get_needle_length(void) { return m_needle_length; }
    static double get_needle_radius(void) { return m_needle_radius; }

    // Needle material properties 
    static double get_needle_density(void) { return m_needle_density; }
    static double get_needle_elasticity(void) { return m_needle_young_modulus; }
    static double get_needle_shear_elasticity(void) { return m_needle_shear_modulus; }

protected:
    // Handle geometry
    static constexpr double m_handle_radius = 1.7e-2; // Handle radius (m) 
    static constexpr double m_handle_length = 1.53e-1; // Handle height (m)
    static constexpr int m_side_count = 20; // Side numbers for cylinder
    static constexpr double m_handle_mass = 9e-2; // Handle mass (kg)

    // Needle geometry
    static constexpr double m_needle_lz = 7.67e-3; // Needle offset z direction (m)
    static constexpr double m_needle_lx = 7.65e-2; // Needle offset x direction (m)
    static constexpr double m_needle_length = 2.623e-1; // Needle length (m)
    static constexpr double m_needle_radius = 6.35e-4; // Needle radius (m)
    
    // Needle material properties 
    static constexpr double m_needle_density = 7850; // Stainless steel density (kg/m^2)
    static constexpr double m_needle_young_modulus = 2e11; // Stainless steel Young Modulus (N/m^2)
    static constexpr double m_needle_shear_modulus = 7.93e10; // Stainless steel Shear Modulus (N/m^2)

};
