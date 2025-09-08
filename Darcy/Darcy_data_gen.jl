using Distributed
@everywhere using NPZ
@everywhere using Interpolations 
@everywhere using Gridap 
@everywhere using Random
@everywhere using Distributions
@everywhere using PyPlot

@everywhere include("Darcy_solver.jl")

@everywhere function g_(ne::Int64, porder::Int64, Xi::Float64, Xf::Float64, Yi::Float64, Yf::Float64, a::Array{Float64, 2})
    sol = darcy_solve(ne, porder, Xi, Xf, Yi, Yf, a)
    return sol
end



porder = 1
ne = 100
N_data = 1000
N_θ = 16
seed = 0
Random.seed!(seed)

θf = rand(Normal(0,1), 5000*N_data*N_θ)
θf = filter(x -> abs(x) <= 4, θf)
@assert size(θf)[1] >= N_data*N_θ
θf = θf[1:(N_data*N_θ)]
θf = reshape(θf, N_data, N_θ, 1)

seq_pairs = compute_seq_pairs(100)
θff = 0.5 * ones(Float64, N_θ)

a_field = ones(Float64, ne+1, ne+1, N_data)
u_field = ones(Float64, ne+1, ne+1, N_data)

Xi, Xf = 0.0, 1.0
Yi, Yf = Xi, Xf

X = LinRange(Xi, Xf, ne+1)
Y = LinRange(Yi, Yf, ne+1)

xx, yy = MeshGrid(collect(X), collect(Y))

for i = 1:N_data
    a_field[:, :, i] .= reshape(a_func_random(xx, yy, θf[i, :], seq_pairs), (ne+1, ne+1))
end

params = [(ne, porder, Xi, Xf, Yi, Yf, a_field[:, :, i]) for i = 1:N_data]

@everywhere params = $params

sols = pmap(param -> g_(param...), params)

for i = 1:N_data
    u_field[:, :, i] .= sols[i]
end
# a2 = sin.(pi * xx) .* sin.(pi * yy)


fig, ax = subplots(ncols=2, figsize=(10,4))
im1 = ax[1].pcolormesh(xx, yy, a_field[:, :, 1], shading="gouraud")
im2 = ax[2].pcolormesh(xx, yy, u_field[:, :, 1], shading="gouraud")
fig.colorbar(im1, ax=ax[1])
fig.colorbar(im2, ax=ax[2])
fig.tight_layout()
savefig("./data_demo.pdf")



npzwrite("./data/darcy_rf_coef_K_$(N_θ)_s$(ne).npy", θf)
npzwrite("./data/darcy_input_a_K_$(N_θ)_s$(ne).npy", a_field)
npzwrite("./data/darcy_output_u_K_$(N_θ)_s$(ne).npy", u_field)




