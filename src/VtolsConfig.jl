eth_vtol_param = Dict(
    "gravity" => 9.81,
    "air_density" => 1.2, # km/m^3 air density 
    "prop_disk" => 0.0133, # 133 cm^2 (inserted in in m^2) propeller disk area
    "kp1" => 1e-4, # Aerodynamic pitching moment. Parameter kp1 is learned during flight. #For stability, the moment must act against the angle of attack.
    # wing and flap caracteristics
    "b_pitch" => 7.46e-6, # Influence of wing airflow on pitching in Nm/(m/s)^2
    "b_yaw" => 4.12e-6, # Influence of wing airflow on yawing in Nm/(m/s)^2
    "b_roll" => 3.19e-6, # Influence of wing airflow on rolling in Nm/(m/s)^2
    "c_pitch" => 2.18e-4, # Influence of the deflected flap airflow through the adjustable flaps on pitching in Nm/(m/s)^2/rad
    "c_roll" => 3.18e-4, # Influence of the deflected flap airflow through the adjustable flaps on rolling in Nm/(m/s)^2/rad
    # Motor Parameters
    "torque_to_thrust" => 8.72e-3, # in Nm/N 
    "prop_distance" => 0.14, # 14 cm Propeller offset along the y_B axis
    "rotation_damping" => [ 0.0; 0.0; 0.0],
    # Estimated aerodynamic parameters
    "kl1" => 1e-1,
    "kl2" => 1e-1,
    "kl3" => 0.0,
    "kd1" => 1e-4,
    "kd2" => 1e-4,
    "kd3" => 0.0,
    "f_min" => 0.1, # N thrust on rotors
    "f_max" => 1.2,
    "δ_min" => -0.87,
    "δ_max" => 0.79,
    # Rigid Body
    "mass" => 0.150,
    "inertia" => [ 4.62e-4 0.0 0.0; 0.0 2.32e-3 0.0 ; 0.0 0.0 1.87e-3],
    "inertia_inv" => inv([ 4.62e-4 0.0 0.0; 0.0 2.32e-3 0.0 ; 0.0 0.0 1.87e-3]),
    "CoM" => [ 0.0; 0.0; 0.0],
    "Cw" => 1.15,
)

crazyflie_param = Dict(
    "gravity" => 9.81,
    # Motor Parameters
    "rotation_damping" => [ 1.0*1e-8; 1.0*1e-8; 1.0*1e-8],
    "translation_damping" => [ 9.0*1e-7; 9.0*1e-7; 9.0*1e-7],
    # Rigid Body
    "mass" => 0.028,
    "inertia" => ([16.571710 0.830806 0.718277; 0.830806 16.655602 1.800197; 0.718277 1.800197 29.261652] .* (1e-6)),
    "inertia_inv" => inv([16.571710 0.830806 0.718277; 0.830806 16.655602 1.800197; 0.718277 1.800197 29.261652] .* (1e-6)),
    "CoM" => [ 0.0; 0.0; 0.0],
)
