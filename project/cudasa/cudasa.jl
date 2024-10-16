using Mydraftcodes
using CairoMakie
using CUDA
using Mydraftcodes:track_equilibration_collective_temperature_gpu!, track_equilibration_collective_temperature_cpu!
using Mydraftcodes:random_state, SimulatedAnnealingHamiltonian, calculate_energy
using Mydraftcodes:HeatBath


function collective_temperature_gpu()
    width=7
    depth=7
    gauss_width=1.0
    λ = 1.5
    CUDA.device!(4)

    scan_time = Vector(1000:100:40000)
    output = []
    nbatch = 100000
    for t in scan_time
        sa = SimulatedAnnealingHamiltonian(width, depth)
        state = CuArray(random_state(sa, nbatch))
        @time track_equilibration_collective_temperature_gpu!(HeatBath(), sa, state, 10.0, t; accelerate_flip = true)
        
        cpu_state = Array(state)
        state_energy = [calculate_energy(sa, cpu_state, fill(1.0, nbatch), i) for i in 1:nbatch]
        success = count(x -> x == 0, state_energy)
        @info "time = $t, success time = $success"
        filepath = joinpath(@__DIR__, "data_tsweep/W=$(width)_D=$(depth)_t=$(t)_E=$(λ).txt")
        open(filepath,"w") do file
            println(file, success)
        end
        push!(output, success)
    end
end

function readout()
    filepath = joinpath(@__DIR__, "data_tsweep")
    txt_files = readdir(filepath)
    T_data = []
    success_data = []
    for file in txt_files
        filename = split(file, ".")[1]
        parts = split(filename, "_")
        t_value = 0
        for part in parts
            if startswith(part, "t=")
                t_value = parse(Int, split(part, "=")[2])
                break
            end
        end
        success_time = open(joinpath(filepath, file)) do f
            parse(Float64, readline(f))
        end
        push!(T_data, t_value)
        push!(success_data, success_time)
    end
    return Float64.(T_data), Float64.(success_data)
end

function plot_data()
    T_data, success_data = readout()
    error_rate = 1 .- success_data ./ 1000
    f = Figure()
    ax = Axis(f[1, 1])
    scatter!(T_data, error_rate)
    f
end

function __main__()
    collective_temperature_gpu()
    plot_data()
end