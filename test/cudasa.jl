using Mydraftcodes, Test
using CairoMakie
using Mydraftcodes:track_equilibration_collective_temperature_gpu!, track_equilibration_collective_temperature_cpu!
using Mydraftcodes:random_state, SimulatedAnnealingHamiltonian, calculate_energy
using Mydraftcodes:HeatBath

@testset "collective_temperature_cpu" begin
    width=7
    depth=7
    gauss_width=1.0
    λ = 1.5

    scan_time = Vector(1000:500:2000)
    output = []
    nbatch = 1000
    for t in scan_time
        sa = SimulatedAnnealingHamiltonian(width, depth)
        state = (random_state(sa, nbatch))
        @time track_equilibration_collective_temperature_cpu!(HeatBath(), sa, state, 10.0, t; accelerate_flip = true)
        
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

