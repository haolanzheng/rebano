using LinearAlgebra
using Gridap
using Interpolations

function MeshGrid(x::Vector{Float64}, y::Vector{Float64})
    nx, ny = length(x), length(y)
    X = zeros(Float64, ny, nx)
    Y = zeros(Float64, ny, nx)
    for i = 1:ny
        X[i, :] .= x
    end
    for i = 1:nx
        Y[:, i] .= y
    end

    return X, Y

end


function compute_seq_pairs(N_KL::Int64) 
    seq_pairs = zeros(Int64, N_KL, 2)
    trunc_Nx = trunc(Int64, sqrt(2*N_KL)) + 1
    
    include_00 = false
    seq_pairs = zeros(Int64, (trunc_Nx+1)^2 - 1 + include_00, 2)
    seq_pairs_mag = zeros(Int64, (trunc_Nx+1)^2 - 1 + include_00)
    
    seq_pairs_i = 0
    for i = 0:trunc_Nx
        for j = 0:trunc_Nx
            if (i == 0 && j ==0 && ~include_00)
                continue
            end
            seq_pairs_i += 1
            seq_pairs[seq_pairs_i, :] .= i, j
            seq_pairs_mag[seq_pairs_i] = i^2 + j^2
        end
    end
    
    seq_pairs = seq_pairs[sortperm(seq_pairs_mag), :]
    return seq_pairs[1:N_KL, :]
end

function a_func_random(x1::Array{Float64, 2}, x2::Array{Float64, 2}, θ::Array{Float64, 1}, seq_pairs::Array{Int64, 2}; d::Float64=2.0, τ::Float64=3.0) 
    
    N_KL = length(θ)
    
    a = zeros(Float64, size(x1)[1], size(x2)[1])
    
    for i = 1:N_KL
        λ = (pi^2*(seq_pairs[i, 1]^2 + seq_pairs[i, 2]^2) + τ^2)^(-d)
        
        if (seq_pairs[i, 1] == 0 && seq_pairs[i, 2] == 0)
            a += θ[i] * sqrt(λ)
        elseif (seq_pairs[i, 1] == 0)
            a += θ[i] * sqrt(λ) * sqrt(2)*cos.(pi * (seq_pairs[i, 2]*x2))
        elseif (seq_pairs[i, 2] == 0)
            a += θ[i] * sqrt(λ) * sqrt(2)*cos.(pi * (seq_pairs[i, 1]*x1))
        else
            a += θ[i] * sqrt(λ) * 2*cos.(pi * (seq_pairs[i, 1]*x1)) .*  cos.(pi * (seq_pairs[i, 2]*x2))
        end
    end
    a .= ifelse.(a .> 0.05, 12.0, ifelse.(a .< -0.05, 3.0, 6.0))

    return a
end

function a_func_random(x1::Float64, x2::Float64, θ::Array{Float64, 1}, seq_pairs::Array{Int64, 2}; d::Float64=2.0, τ::Float64=3.0) 
    
    N_KL = length(θ)
    
    a = 0.0
    
    for i = 1:N_KL
        λ = (pi^2*(seq_pairs[i, 1]^2 + seq_pairs[i, 2]^2) + τ^2)^(-d)
        
        if (seq_pairs[i, 1] == 0 && seq_pairs[i, 2] == 0)
            a += θ[i] * sqrt(λ)
        elseif (seq_pairs[i, 1] == 0)
            a += θ[i] * sqrt(λ) * sqrt(2)*cos.(pi * (seq_pairs[i, 2]*x2))
        elseif (seq_pairs[i, 2] == 0)
            a += θ[i] * sqrt(λ) * sqrt(2)*cos.(pi * (seq_pairs[i, 1]*x1))
        else
            a += θ[i] * sqrt(λ) * 2*cos.(pi * (seq_pairs[i, 1]*x1)) .*  cos.(pi * (seq_pairs[i, 2]*x2))
        end
    end
    a .= ifelse.(a .> 0.0, 12.0, 3.0)

    return a
end

function darcy_solve(ne::Int64, porder::Int64, Xi::Float64, Xf::Float64, Yi::Float64, Yf::Float64, a_field::Array{Float64, 2})
    X = LinRange(Xi, Xf, ne+1)
    Y = LinRange(Yi, Yf, ne+1)

    xx, yy = MeshGrid(collect(X), collect(Y))

    # θ_field = npzread("/groups/esm/dzhuang/Operator-Learning/data/Random_Helmholtz_high_theta_100.npy") 
    u_field = zeros(Float64, ne+1, ne+1)

    nex, ney = ne, ne
    partition = (nex, ney)

    domain = (Xi, Xf, Yi, Yf)
    model = CartesianDiscreteModel(domain, partition)

    labels = get_face_labeling(model)

    add_tag_from_tags!(labels, "DirichletBoundary", [1, 2, 3, 4, 5, 6, 7, 8])

    f(x) = 1.0 # 2pi^2*sin(pi*x[1]) * sin(pi*x[2])

    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2porder)

    # println("number of cells used : ", num_cells(model))


    reffe = ReferenceFE(lagrangian, Float64, porder)
    V     = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags = ["DirichletBoundary"])
    U     = TrialFESpace(V)

    coords = zeros(Float64, (nex+1)*(ney+1), 2)

    xx, yy = MeshGrid(collect(X), collect(Y))
    coord_x = reshape(xx, :)
    coord_y = reshape(yy, :)
    coords  = hcat(coord_x, coord_y)

    
    a_data = a_field
    itp = Interpolations.interpolate(a_data, BSpline(Cubic()), OnGrid())
    a_interp = Interpolations.scale(itp, X, Y)
    a_f(x) = a_interp(x[1], x[2])

    a(u, v) = ∫(a_f * ∇(u) ⋅ ∇(v) )*dΩ
    l(v) = ∫( f * v) * dΩ


    # Assemble the finite element operator with the Dirichlet condition
    op = AffineFEOperator(a, l, U, V)
    solver = LUSolver()
    uh = solve(solver, op)
    
    sol = [evaluate(uh, Point(x, y)) for (x, y) in zip(coords[:, 1], coords[:, 2])]
    sol = reshape(sol, nex+1, ney+1)

    u_field[:, :] .= sol 

    
    u_field
end