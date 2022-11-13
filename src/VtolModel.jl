module VtolModel

include("RigidBodies.jl");
using .RigidBodies;

export vtol_model, init_eth_vtol_model, MC_model, init_MC_vtol_model

function init_eth_vtol_model(vtol_params)

    function eth_vtol_model(x_W, v_B, Q_W, ω_B, airspeed_x_B, t, Δt, action)

        torque_B, force_B = vtol_model(v_B, action, vtol_params)

        # set the next forces for the rigid body simulation
        x_W, v_B, Q_W, ω_B, t = rigid_body_quaternion(torque_B, force_B, x_W, v_B, Q_W, ω_B, t, Δt, vtol_params)


        return x_W, v_B, Q_W, ω_B, t
    end

    return eth_vtol_model
end;



function init_MC_vtol_model(vtol_params)

    function MC_vtol_model(x_W, v_B, Q_W, ω_B, airspeed_x_B, t, Δt, action)

        torque_B, force_B = MC_model(v_B, action, vtol_params)

        # set the next forces for the rigid body simulation
        x_W, v_B, Q_W, ω_B, t = rigid_body_quaternion(torque_B, force_B, x_W, v_B, Q_W, ω_B, t, Δt, vtol_params)

        
        return x_W, v_B, Q_W, ω_B, t
    end

    return MC_vtol_model
end;




function MC_model(v_B::Vector{Float64}, actions_PWM::Vector{Float64}, param)

    actions = [(actions_PWM[1] -param["f_shift"])*param["f_scale"];
               (actions_PWM[2] -param["f_shift"])*param["f_scale"]; 
               (actions_PWM[3] -param["δ_r_shift"])*param["δ_scale"];
               (actions_PWM[4] -param["δ_l_shift"])*(-1.0)*param["δ_scale"]]

    # Reference airspeed over the flaps along the x_B axis
    airspeed_x_l = actions[2];
    airspeed_x_r = actions[1];

    torque_pitch = param["c_pitch"] * ( actions[4] * airspeed_x_l + actions[3] * airspeed_x_r);
    torque_roll = param["c_roll"] *  (actions[3] * airspeed_x_r - actions[4] * airspeed_x_l);

    angle_of_attack = atan(v_B[3] , v_B[1]); # angle of attack
    airspeed = sqrt(v_B[3]^2 +  v_B[1]^2); # velocity of the aircraft relative to the surrounding air

    m_pitch = param["kp1"] * sin(angle_of_attack) * airspeed^2;



    torque_yaw = (param["prop_distance"] - param["prop_shift"]) * actions[2] - (param["prop_distance"] + param["prop_shift"]) * actions[1];
    torque_B = [torque_roll ; torque_pitch + m_pitch ; torque_yaw]           
    force_B = [ actions[2] + actions[1]; 0.0; 0.0]


    return torque_B, force_B
end



    # Implementation of the paper "A global controller for flying wing tailsitter vehicles" in Julia
    # https://www.flyingmachinearena.ethz.ch/wp-content/uploads/ritzIEEE17.pdf
    """
        vtol_model(v_B::Array{Real,1}, actions::Array{Real,1}, param::Dict{String, Real})
    
    calculates general parameters

    Angle of attack ``α = arctan2(-v_z^B, v_x^B)``,

    Airspeed ``ν = \\sqrt{(-v_z^B)^2 + (v_x^B)^2}``,
    
    limits the actuator inputs,

    ``f_{min} <= f_l, f_r <= f_{max}`` for the thrust,

    `` δ_{min} <= δ_l, δ_r <= δ_{max}`` for the flaps,

    and combines the different forces

    ``τ^B = `` aerodynamic_torque_model ``+`` propeller_torque_model
    
    ``f^B = `` aerodynamic_force_model ``+`` propeller_force_model

    """
    function vtol_model(v_B, actions, param)
        # limits for motors and flaps
        actions = [ max(param["f_min"], min(actions[1], param["f_max"])),
                    max(param["f_min"], min(actions[2], param["f_max"])),
                    0,
                    0]
                    #max(param["δ_min"], min(actions[3], param["δ_max"])),
                    #max(param["δ_min"], min(actions[4], param["δ_max"]))]


        angle_of_attack = atan(v_B[3] , v_B[1]); # angle of attack
        airspeed = sqrt(v_B[3]^2 +  v_B[1]^2); # velocity of the aircraft relative to the surrounding air

        torque_B = aerodynamic_torque_model(v_B, actions, angle_of_attack, airspeed, param) .+
                   propeller_torque_model(actions, param);


        force_B = aerodynamic_force_model(actions, angle_of_attack, airspeed, param) .+
                  propeller_force_model(actions);

        return torque_B, force_B
    end



    # ----------------------------------- TORQUE MODEL -----------------------------------
    # Aerodynamic torque model - The body of a flying wing causes an aerodynamic pitching moment around the wing axis
    """
        aerodynamic_torque_model(v_B::Array{Real,1}, actions::Array{Real,1}, angle_of_attack::Real, airspeed::Real, param::Dict{String, Real})

    Calculate the torques caused by the aerodynamics of the drone.

    Reference airspeed over the `` s \\in \\{left, right\\} `` flap alon the body `` x `` axis.

    `` v^B_{s,x} = \\sqrt{ \\frac{2 f_s}{ρ d} + max(0, v^B_z)^2} ``

    `` ρ `` is the air density, `` d `` the propeller disk area, `` f_s `` the force set for the left / right rotor and `` v^b_x `` the body velocity in ``x`` direction.

    Total reference airspeed over the flaps

    `` v_s = \\sqrt{ (v_y^B)^2 + (v^B_{s,x})^2} ``

    Pitching moment

    `` m_{pitch} = k_{p1} sin(α) v^2  ``

    `` α `` is the angle of attack, ``v ``the total velocitie and ``k_{p1}`` a learnd parameter.

    Torques acting on the body generated by aerodynamics

    `` m^B_x = (b_x + c_x δ_l) v_l^2 + (b_x + c_x δ_r) v_r^2 + m_{pitch} ``

    `` m^B_y = b_y v_l^2 - b_y v_r^2 ``

    `` m^B_z = (b_z + c_z δ_l) v^2_l - (b_z + c_z δ_r) v^2_r ``

    ``δ_l`` and ``δ_r`` are the flap angles, the ``b`` and ``c`` values are aerodynamic coefficients.

    The model comes from the paper: "A global controller for flying wing tailsitter vehicls, 2017".
    """
    function aerodynamic_torque_model(v_B, actions, angle_of_attack, airspeed, param)


        # Reference airspeed over the flaps along the x_B axis
        airspeed_x_l = sqrt(((2*actions[2]) / (param["air_density"] * param["prop_disk"])) + max(0, v_B[1])^2);
        airspeed_x_r = sqrt(((2*actions[1]) / (param["air_density"] * param["prop_disk"])) + max(0, v_B[1])^2);

        # Total reference airspeed over the flaps
        airspeed_l = sqrt(v_B[3]^2 + airspeed_x_l^2);
        airspeed_r = sqrt(v_B[3]^2 + airspeed_x_r^2);

        # Parameterized aerodynamic functions. 
        m_pitch = param["kp1"] * sin(angle_of_attack) * airspeed^2;

        torque_pitch = (param["b_pitch"] + param["c_pitch"] * actions[4]) * airspeed_l^2 + (param["b_pitch"] + param["c_pitch"] * actions[3]) * airspeed_r^2 + m_pitch;
        torque_roll = (param["b_roll"] + param["c_roll"] * actions[3]) * airspeed_r^2 - (param["b_roll"] + param["c_roll"] * actions[4]) * airspeed_l^2;
        # TODO: What is the idea behind this part?
        torque_yaw = param["b_yaw"] * airspeed_r^2 - param["b_yaw"] * airspeed_l^2; # TODO: MATLAB IMPLEMENTATION + v_body_mps(1)^2 * .0025; ???
        
        # aerodynamic torque around the body axes
        return [torque_roll, torque_pitch, torque_yaw];
    end

    """
        propeller_torque_model( actions::Array{Real,1}, param::Dict{String, Real} )

    Calculate the torque caused directly and only by the propellers.

    ``τ_{pitch} = 0``

    ``τ_{roll} = κ \\cdot (f_r - f_l)``

    ``τ_{yaw} = l \\cdot (f_l - f_r)``
    
    with ``l = `` Propeller lever arm and ``κ = `` torque to thrust
    """
    function propeller_torque_model(actions, param)

        torque_pitch = 0; 
        # Note that the left propeller rotates counter-clockwise, and the right propeller rotates clockwise.
        # TODO: Rotation direction is determined when looking from the front (Not the Drone perspective)???
        torque_roll = param["torque_to_thrust"] * (actions[1] - actions[2]);
        torque_yaw = param["prop_distance"] * (actions[2] - actions[1]);

        # propeller torque around the body axes
        return [torque_roll, torque_pitch, torque_yaw];

    end


    # ----------------------------------- FORCE MODEL -----------------------------------

    # Aerodynamic force model
    """
        aerodynamic_force_model(actions::Array{Real,1}, aoa::Real, airspeed::Real, param::Dict{String, Real})

    Calculate the force caused by the aerodynamics of the drone.
    Assumption 1: The lateral (y direction) aerodynamic force acting on the vehicle is negligible.
    Assumption 2: the aerodynamic force is a function of the angle of attack, the reference airspeed, and the average propeller force.

    ``f_{x} = -f_{drag}``

    ``f_{y} = 0``

    ``f_{z} = -f_{lift}``

    with drone parameters ``k_{l1}, k_{l2}, k_{l3}, k_{d1}, k_{d2}, k_{d3}``, the angle of attack ``α``, the airspeed ``ν`` the average propeller force ``f_{lr} = \\frac{f_r + f_l}{2}`` and

    ``f_{lift} = (k_{l1} * sin(α) * cos(α)^2 + k_{l2} * sin(α)^3) * ν^2 + k_{l3} * f_{kl}``,

    ``f_{drag} = (k_{d1} * sin(α)^2 * cos(α) + k_{d2} * cos(α)) * ν^2 + k_{d3} * f_{kl}``.

    The model comes from the paper: "A global controller for flying wing tailsitter vehicls, 2017".
    """
    function aerodynamic_force_model(actions, aoa, airspeed, param)

        avg_prop_force = (actions[2] + actions[1])/2.0;

        f_lift = ( param["kl1"] * sin(aoa) * cos(aoa)^2 + param["kl2"] * sin(aoa)^3 )*airspeed^2 + param["kl3"]*avg_prop_force
        f_drag = ( param["kd1"] * sin(aoa)^2 * cos(aoa) + param["kd2"] * cos(aoa) )  *airspeed^2 + param["kd3"]*avg_prop_force
        
        # TODO: chose implementation
        # Implementation Leon
        #f_lift = sin(aoa)*1.15*param["face_z"]*param["air_density"]*airspeed^2 /2;
        #f_drag = cos(aoa)*1.4*param["face_x"]*param["air_density"]*airspeed^2 /2;

        force_x = -f_drag;
        force_y = 0;
        force_z = -f_lift;

        # aerodynamic force along the body axes
        return [force_x, force_y, force_z];
    end


    """
        aerodynamic_force_model_wing_surface(actions::Array{Real,1}, aoa::Real, airspeed::Real, param::Dict{String, Real}, lateral_airspeed::Float)

    Calculate the force caused by the aerodynamics of the drone based on the wing surface.

    ``f_{x} = -f_{drag}``

    ``f_{y} = -``lateral_airspeed``^2 * 0.05`` Where does this idea come from? Regardless of the wind direction, a force to one side?

    ``f_{z} = -f_{lift}``

    with drone parameters ``x_{surface}, z_{surface}``, the angle of attack ``α``, the airspeed ``ν``, the air density ``ρ`` and

    ``f_{lift} = \\frac{ sin(α) * 1.15 * z_{surface} * ρ * ν^2}{2}``,

    ``f_{drag} = \\frac{ cos(α) * 1.4 * x_{surface} * ρ * ν^2}{2}``.

    The model comes from the Matlab implementation of Leon Sievers, 2022. Or the Small Unmanned Aircraft book.
    """
function aerodynamic_force_model_wing_surface(actions, aoa, airspeed, param, lateral_airspeed)

    # TODO: chose implementation
    # Implementation Leon
    f_lift = sin(aoa)*1.15*param["face_z"]*param["air_density"]*airspeed^2 /2;
    f_drag = cos(aoa)*1.4*param["face_x"]*param["air_density"]*airspeed^2 /2;

    force_x = -f_drag;
    force_y = 0;#-lateral_airspeed^2 * 0.05; 
    force_z = -f_lift;

    # aerodynamic force along the body axes
    return [force_x, force_y, force_z];
end


    # Propeller force model / Propulsion force model
    """
        propeller_force_model(actions::Array{Real,1})

    Calculate the force caused directly and only by the propellers.

    ``f_{x} = f_r + f_l``

    ``f_{y} = 0``

    ``f_{z} = 0``
    """
    function propeller_force_model(actions)

        force_x = actions[2] + actions[1];
        force_y = 0;
        force_z = 0;

        # propeller force along the body axes
        return [force_x, force_y, force_z];
    end
end # end of module 

