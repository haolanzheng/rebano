using Distributed
@everywhere using LinearAlgebra
@everywhere using Distributions
@everywhere using Random
@everywhere using SparseArrays
@everywhere using NPZ

@everywhere include("./Navier-Stokes-Force-Sol.jl")

@everywhere function g_(x::Matrix{Float64}, ω0::Matrix{Float64}, ν::Float64, N_t::Int64, T::Float64)
    return NS_Solver(x, ω0; ν = ν, N_t = N_t, T = T)
end


function Data_Generate()
    N_data = 1000 #40000
    ν = 0.025                                      # viscosity
    N, L = 100, 2*pi                                 # resolution and domain size 
    N_t = 5000 #10000;     
    nt_sub = 100                                # time step
    T = 10.0;                                        # final time

    d=4.0
    τ=3.0
    # The forcing has N_θ terms
    N_θ = 16
    seq_pairs = Compute_Seq_Pairs(100)

    Random.seed!(0);
    # θf = rand(Normal(0,1), N_data+1, N_θ, 1)
    θf = rand(Normal(0,1), 5000*(N_data+1)*N_θ)
    θf = filter(x -> abs(x) <= 4, θf)
    @assert size(θf)[1] >= (N_data+1)*N_θ
    θf = θf[1:((N_data+1)*N_θ)]
    θf = reshape(θf, N_data+1, N_θ, 1)
    
    curl_f = zeros(N, N, N_data)
    # f = zeros(N, N, 2, N_data)
    for i = 1:N_data
    	# 2*
        curl_f[:,:, i] .= generate_ω0(L, N, θf[i,:], seq_pairs, d, τ)
        # f[:, :, 1, i] .= generate_ω0(L, N, θf[i,:,1], seq_pairs, d, τ)
        # f[:, :, 2, i] .= generate_ω0(L, N, θf[i,:,2], seq_pairs, d, τ)
    end

    θω = θf[end, :]
    ω0 = generate_ω0(L, N, θω, seq_pairs, d, τ)
    # ω0 = npzread("./data/Random_NS_trunc_omega0_8_100.npy")
    # vel0 = zeros(N, N ,2)
    # θu = rand(Normal(0,1), N_θ, 2)
    # vel0[:, :, 1] .= generate_ω0(L, N, θu[:, 1], seq_pairs, d, τ)
    # vel0[:, :, 2] .= generate_ω0(L, N, θu[:, 2], seq_pairs, d, τ)

    
    params = [(curl_f[:, :, i], ω0, ν, N_t, T)  for i in 1:N_data]
    
    @everywhere params = $params
    ω_tuple = pmap(param -> g_(param...), params) # Outer dim is params iterator

    ω_field = zeros(nt_sub+1, N, N, N_data)
    for i = 1:N_data
        ω_field[:, :, :, i] = ω_tuple[i]
    end
    K = N_θ
    npzwrite("./data/ns_rf_coef_K_$(K)_s$(N).npy",  θf)
    npzwrite("ns_omega0_K_$(K)_s$(N).npy", ω0)
    npzwrite("./data/ns_output_omega_K_$(K)_s$(N).npy",  ω_field)
    npzwrite("./data/ns_input_curl_f_K_$(K)_s$(N).npy", curl_f)
    
    # npzwrite("Random_NS_f_$(N_θ)_$(N).npy", f)
    # npzwrite("Random_NS_init_vel_$(N_θ)_$(N).npy", vel0)
end

Data_Generate()