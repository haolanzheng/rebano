using LinearAlgebra 
using SparseArrays
using Random 
using Distributions 
using NPZ 

function poisson_sol()
    K = 10
    N_data = 1000

    L = 1.0 
    Nx = 128
    d = 2.0
    τ = 1.0

    dx = L/Nx
    xx = vec(0:dx:L*(1.0 - 1.0/Nx))

    Random.seed!(0);
    rf_coef = rand(Normal(0,1), 5000*K)
    rf_coef = filter(x -> abs(x) <= 4, rf_coef)
    @assert size(rf_coef)[1] >= N_data*K
    rf_coef = rf_coef[1:N_data*K]
    rf_coef = reshape(rf_coef, K, N_data)
    

    fk_hat = zeros(Float64, Nx, K)
    uk_sol = zeros(Float64, Nx, K)
    eigen_val = [(i^2 + τ^2)^(-d/2) for i in 1:K] 
    k = reshape(1:K, 1, :)
    kx = xx * k  
    @time begin
    norm_sin = sqrt(2)*sin.(pi*kx)
    for i=1:K 
        fk_hat[:, i] .= eigen_val[i] * norm_sin[:, i]
        uk_sol[:, i] .= eigen_val[i]/((i*pi)^2) * norm_sin[:, i]
    end 

    f_hat = fk_hat * rf_coef
    u_sol = uk_sol * rf_coef
    end
    npzwrite("./data/poisson1d_rf_coef_K_$(K)_s$(Nx).npy", rf_coef)
    npzwrite("./data/poisson1d_input_f_K_$(K)_s$(Nx).npy", f_hat)
    npzwrite("./data/poisson1d_output_u_K_$(K)_s$(Nx).npy", u_sol)

end

poisson_sol()