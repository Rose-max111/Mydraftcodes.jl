struct SimulatedAnnealingHamiltonian
    n::Int # number of atoms per layer
    m::Int # number of full layers (count the last layer!)
end

abstract type TempcomputeRule end
struct Gaussiantype <: TempcomputeRule end
struct Exponentialtype <: TempcomputeRule end

abstract type TransitionRule end
struct HeatBath <: TransitionRule end
struct Metropolis <: TransitionRule end
prob_accept(::Metropolis, Temp, ΔE::T) where T<:Real = ΔE < 0 ? 1.0 : exp(- (ΔE) / Temp)
prob_accept(::HeatBath, Temp, ΔE::Real) = inv(1 + exp(ΔE / Temp))

natom(sa::SimulatedAnnealingHamiltonian) = sa.n * sa.m
atoms(sa::SimulatedAnnealingHamiltonian) = Base.OneTo(natom(sa))
function random_state(sa::SimulatedAnnealingHamiltonian, nbatch::Integer)
    return rand(Bool, natom(sa), nbatch)
end
hasparent(sa::SimulatedAnnealingHamiltonian, node::Integer) = node > sa.n

rule110(p, q, r) = (q + r + q*r + p*q*r) % 2

# evaluate the energy of the i-th gadget (involving atoms i and its parents)
function evaluate_parent(sa::SimulatedAnnealingHamiltonian, state::AbstractMatrix, energy_gradient::AbstractArray, inode::Integer, ibatch::Integer)
    i, j = CartesianIndices((sa.n, sa.m))[inode].I
    idp = parent_nodes(sa, inode)
    trueoutput = @inbounds rule110(state[idp[1], ibatch], state[idp[2], ibatch], state[idp[3], ibatch])
    return @inbounds (trueoutput ⊻ state[inode, ibatch]) * (energy_gradient[ibatch] ^ (sa.m - j))
end
function calculate_energy(sa::SimulatedAnnealingHamiltonian, state::AbstractMatrix, energy_gradient::AbstractArray, ibatch::Integer)
    return sum(i->evaluate_parent(sa, state, energy_gradient, i, ibatch), sa.n+1:natom(sa))
end

function parent_nodes(sa::SimulatedAnnealingHamiltonian, node::Integer)
    n = sa.n
    i, j = CartesianIndices((n, sa.m))[node].I
    lis = LinearIndices((n, sa.m))
    @inbounds (
        lis[mod1(i-1, n), j-1],  # periodic boundary condition
        lis[i, j-1],
        lis[mod1(i+1, n), j-1],
    )
end

function child_nodes(sa::SimulatedAnnealingHamiltonian, node::Integer)
    n = sa.n
    i, j = CartesianIndices((n, sa.m))[node].I
    lis = LinearIndices((n, sa.m))
    @inbounds (
        lis[mod1(i-1, n), j+1],  # periodic boundary condition
        lis[mod1(i, n), j+1],
        lis[mod1(i+1, n), j+1],
    )
end

# step for cpu test
function step!(rule::TransitionRule, sa::SimulatedAnnealingHamiltonian, state::AbstractMatrix, energy_gradient::AbstractArray, Temp, node=nothing)
    for ibatch in 1:size(state, 2)
        step_kernel!(rule, sa, state, energy_gradient, Temp, ibatch, node)
    end
    state
end
# step for non-parallel-flipping gpu test
function step!(rule::TransitionRule, sa::SimulatedAnnealingHamiltonian, state::CuMatrix, energy_gradient::AbstractArray, Temp, node=nothing)
    @inline function kernel(rule::TransitionRule, sa::SimulatedAnnealingHamiltonian, state::AbstractMatrix, energy_gradient::AbstractArray, Temp, node=nothing)
        ibatch = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        if ibatch <= size(state, 2)
            step_kernel!(rule, sa, state, energy_gradient, Temp, ibatch, node)
        end
        return nothing
    end
    kernel = @cuda launch=false kernel(rule, sa, state, energy_gradient, Temp, node)
    config = launch_configuration(kernel.fun)
    threads = min(size(state, 2), config.threads)
    blocks = cld(size(state, 2), threads)
    CUDA.@sync kernel(rule, sa, state, energy_gradient, Temp, node; threads, blocks)
    state
end

@inline function step_kernel!(rule::TransitionRule, sa::SimulatedAnnealingHamiltonian, state, energy_gradient::AbstractArray, Temp, ibatch::Integer, node=nothing)
    ΔE_with_next_layer = 0
    ΔE_with_previous_layer = 0
    if node === nothing
        node = rand(atoms(sa))
    end
    i, j = CartesianIndices((sa.n, sa.m))[node].I
    if j > 1 # not the first layer
        ΔE_with_previous_layer += energy_gradient[ibatch]^(sa.m - j) - 2 * evaluate_parent(sa, state, energy_gradient, node, ibatch)
    end
    if j < sa.m # not the last layer
        cnodes = child_nodes(sa, node)
        for node in cnodes
            ΔE_with_next_layer -= evaluate_parent(sa, state, energy_gradient, node, ibatch)
        end
        # flip the node@
        @inbounds state[node, ibatch] ⊻= true
        for node in cnodes
            ΔE_with_next_layer += evaluate_parent(sa, state, energy_gradient, node, ibatch)
        end
        @inbounds state[node, ibatch] ⊻= true
    end
    flip_max_prob = 1
    if j == sa.m
        flip_max_prob *= prob_accept(rule, Temp[ibatch][j-1], ΔE_with_previous_layer)
    elseif j == 1
        flip_max_prob *= prob_accept(rule, Temp[ibatch][j], ΔE_with_next_layer)
    else
        flip_max_prob = 1.0 / (1.0 + exp(ΔE_with_previous_layer / Temp[ibatch][j-1] + ΔE_with_next_layer / Temp[ibatch][j]))
    end
    if rand() < flip_max_prob
        @inbounds state[node, ibatch] ⊻= true
        ΔE_with_next_layer
    else
        0
    end
end

function step_parallel!(rule::TransitionRule, sa::SimulatedAnnealingHamiltonian, state::AbstractMatrix, energy_gradient::AbstractArray, Temp, flip_id)
    # @info "flip_id = $flip_id"
    for ibatch in 1:size(state, 2)
        for this_time_flip in flip_id
            step_kernel!(rule, sa, state, energy_gradient, Temp, ibatch, this_time_flip)
        end
    end
    state
end
function step_parallel!(rule::TransitionRule, sa::SimulatedAnnealingHamiltonian, state::CuMatrix, energy_gradient::AbstractArray, Temp, flip_id)
    @inline function kernel(rule::TransitionRule, sa::SimulatedAnnealingHamiltonian, state::AbstractMatrix, energy_gradient::AbstractArray, Temp, flip_id)
        id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        stride = blockDim().x * gridDim().x
        Nx = size(state, 2)
        Ny = length(flip_id)
        cind = CartesianIndices((Nx, Ny))
        for k in id:stride:Nx*Ny
            ibatch = cind[k][1]
            id = cind[k][2]
            step_kernel!(rule, sa, state, energy_gradient, Temp, ibatch, flip_id[id])
        end
        return nothing
    end
    kernel = @cuda launch=false kernel(rule, sa, state, energy_gradient, Temp, flip_id)
    config = launch_configuration(kernel.fun)
    threads = min(size(state, 2) * length(flip_id), config.threads)
    blocks = cld(size(state, 2) * length(flip_id), threads)
    CUDA.@sync kernel(rule, sa, state, energy_gradient, Temp, flip_id; threads, blocks)
    state
end

function get_parallel_flip_id(sa)
    ret = Vector{Vector{Int}}()
    for cnt in 1:6
        temp = Vector{Int}()
        for layer in 1+div(cnt - 1, 3):2:sa.m
            for position in mod1(cnt, 3):3:(sa.n - sa.n % 3)
                push!(temp, LinearIndices((sa.n, sa.m))[position, layer])
            end
        end
        push!(ret, temp)
    end
    if sa.n % 3 >= 1
        push!(ret, Vector(sa.n:2*sa.n:sa.n*sa.m))
        push!(ret, Vector(2*sa.n:2*sa.n:sa.n*sa.m))
    end
    if sa.n % 3 >= 2
        push!(ret, Vector(sa.n-1:2*sa.n:sa.n*sa.m))
        push!(ret, Vector(2*sa.n-1:2*sa.n:sa.n*sa.m))
    end
    return ret
end

function track_equilibration_collective_temperature_cpu!(rule::TransitionRule,
                                        sa::SimulatedAnnealingHamiltonian, 
                                        state::AbstractMatrix,
                                        max_temperature,
                                        annealing_time; accelerate_flip = false)
    each_decrease = (max_temperature - 1e-5) / annealing_time
    now_temperature = max_temperature
    for t in 1:annealing_time
        singlebatch_temp = fill(now_temperature, sa.m-1)
        Temp = fill(singlebatch_temp, size(state, 2))
        if accelerate_flip == false
            for thisatom in 1:natom(sa)
                step!(rule, sa, state, fill(1.0, size(state, 2)), Temp, thisatom)
            end
        else
            flip_list = get_parallel_flip_id(sa)
            for eachflip in flip_list
                step_parallel!(rule, sa, state, fill(1.0, size(state, 2)), Temp, eachflip)
            end
        end
        now_temperature -= each_decrease
    end
end

function track_equilibration_collective_temperature_gpu!(rule::TransitionRule,
                                        sa::SimulatedAnnealingHamiltonian, 
                                        state::AbstractMatrix,
                                        max_temperature,
                                        annealing_time; accelerate_flip = false)
    each_decrease = (max_temperature - 1e-5) / annealing_time
    now_temperature = max_temperature
    for t in 1:annealing_time
        singlebatch_temp = fill(Float32(now_temperature), sa.m-1)
        Temp = CuArray(fill(singlebatch_temp, size(state, 2)))
        if accelerate_flip == false
            for thisatom in 1:natom(sa)
                step!(rule, sa, state, CuArray(fill(1.0f0, size(state, 2))), Temp, thisatom)
            end
        else
            flip_list = get_parallel_flip_id(sa)
            for eachflip in flip_list
                step_parallel!(rule, sa, state, CuArray(fill(1.0f0, size(state, 2))), Temp, CuArray(eachflip))
            end
        end
        now_temperature -= each_decrease
    end
end
