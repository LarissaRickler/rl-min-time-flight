module Utils
using LinearAlgebra;

using Random;
using Distributions;

export calculateAngle, progress, generate_trajectory, visualize_waypoints

function calculateAngle(a::Vector{T}, b::Vector{T}) where T
    return acos(clamp(dot(a,b)/(norm(a)*norm(b)), -1, 1))*sign(b[1])
end


function progress_along_line(gate::Vector{T}, next_gate::Vector{T}, x_W::Vector{T}) where T
    t = dot(x_W - gate, next_gate - gate) / norm(next_gate - gate)^2 
    t = clamp(t, 0, 1)
    phi = gate + t * (next_gate - gate)
    return phi
end


function progress(gates::Vector{Vector{T}}, x_W::Vector{T}) where T
    #Debug: maybe start has to be added to gates ?
    phis = Vector{Vector{T}}(undef, size(gates,1)-1)
    dist = Vector{T}(undef, size(gates,1) - 1)
    for i in eachindex(phis)
        phis[i] = progress_along_line(gates[i], gates[i + 1], x_W)
        dist[i] = norm(x_W - phis[i])
    end

    min_index = argmin(dist)
    return min_index, phis[min_index]
end


function generate_trajectory(num_waypoints::Int) #DEBUG: harder trajectories
    waypoints = Vector{Vector{Float64}}(undef, num_waypoints + 1) 
    waypoints[1] = [0.0, 0.0, 0.0] # zero is first waypoint 
    for i in 1:num_waypoints
        # DEBUG: maybe different ranges
        waypoints[i+1] = waypoints[i] + rand(Uniform(1.5,7),3)
        waypoints[i+1,][2] = 0.0 # TODO: remove for 3d
    end
    return waypoints
end


function visualize_waypoints(waypoints::Vector{Vector{T}}, radius::Real; color=RGBA{Float32}(0.0, 1.0, 0.0, 1.0)) where T
    for i in eachindex(waypoints)
        create_sphere("fixgoal_$i", radius, color=color);
        set_transform("fixgoal_$i", waypoints[i]); 
    end
end


function testRotationMatrix(R)

    determinante = (det(R) ≈ 1)
    #inverse = (transpose(R) == inv(R))
    return determinante# && inverse
end




function projectSO3C(R)
    
    R_sym = transpose(R) * R
    eigval = eigvals(R_sym)
    eigvec = eigvecs(R_sym)
    
    u = reverse(eigvec, dims=2)
    
    sign = 1;
    if (det(R) < 0)
        sign = -1;
    end
    
    dMat = [eigval[3]^(-0.5); eigval[2]^(-0.5); sign*eigval[1]^(-0.5)]
    
    R_SO3 = R * u * Diagonal(dMat) * transpose(u)
    
    if !testRotationMatrix(R_SO3)
        println("ERROR: Rotation not on SO3")
    end
    return R_SO3
end


        # TODO: Docu !!!
        # Projection on to SO3 (Projection onto Essential Space: https://vision.in.tum.de/_media/teaching/ss2016/mvg2016/material/multiviewgeometry5.pdf)
        #F = svd(exponential_map)
        #σ = (F.S[1] + F.S[2]) / 2.0
        #S = [σ, σ, 0.0]
        #A = F.U * Diagonal(S) * F.Vt



denormalize_data(x,mean,std) = (x .* (std .+ 1e-5)) .+ mean;
normalize_data(x, mean, std) = (x .- mean) ./ (std .+ 1e-5);


Quat2Rot(q) = transpose([2*(q[1]^2 + q[2]^2)-1 2*(q[2]*q[3] + q[1]*q[4]) 2*(q[2]*q[4] - q[1]*q[3]);
                     2*(q[2]*q[3] - q[1]*q[4]) 2*(q[1]^2 + q[3]^2)-1 2*(q[3]*q[4] + q[1]*q[2]);
                     2*(q[2]*q[4] + q[1]*q[3]) 2*(q[3]*q[4] - q[1]*q[2]) 2*(q[1]^2 + q[4]^2)-1])
end
