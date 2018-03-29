module MGLDA

struct returns
    Φ_::Array{Float64,2}
    corpus::Vector{String}
    n_components::Tuple{Int,Int}
end

using StatsBase

const α_loc = 0.5
const α_gl  = 0.5
const β     = 0.1
const γ     = 0.5
const a     = 1.0
const b     = 1.0

function fit(X_::Array, corpus::Vector; max_iter::Int=100, n_components::Tuple=(10,30), T::Int=3)
    @assert max_iter > 0
    @assert n_components[1] > 0 && n_components[2] > 0 && T > 0

    cgs(X_, corpus, max_iter, n_components, T)
end

# get parameters for this estimator
function get_params()
    @assert isdefined(params, :corpus)

    return params
end

# return the most probable topn words in topic topicid
function show_topic(topicid::Int; topictype::String="loc", topn::Int=10)
    @assert isdefined(params, :corpus)
    @assert topictype ∈ ["loc", "gl"]
    @assert topn <= length(params.corpus)

    if topictype == "loc"
        @assert 1 <= topicid <= params.n_components[1]
        return [params.corpus[w] for w in sortperm(params.Φ_[topicid,:], rev=true)[1:topn]]
    elseif topictype == "gl"
        @assert 1 <= topicid <= params.n_components[2]
        return [params.corpus[w] for w in sortperm(params.Φ_[params.n_components[1]+topicid,:], rev=true)[1:topn]]
    end
end

# estimate model parameters with collapsed Gibbs sampling
function cgs(X_::Array, corpus::Vector, max_iter::Int, n_components::Tuple, T::Int)
    n_docs  = length(X_)
    n_words = length(corpus)

    r_dsn = [[zeros(Int, N_ds) for (N_ds, X) in X_[d][1]] for d in 1:n_docs]
    z_dsn = [[zeros(Int, N_ds) for (N_ds, X) in X_[d][1]] for d in 1:n_docs]
    v_dsn = [[zeros(Int, N_ds) for (N_ds, X) in X_[d][1]] for d in 1:n_docs]

    N_dsv = [zeros(Int, length(X_[d][1]), length(X_[d][2])) for d in 1:n_docs]

    N_dvr = [zeros(Int, length(X_[d][2]), 2) for d in 1:n_docs]
    N_dv  = [zeros(Int, length(X_[d][2])) for d in 1:n_docs]

    N_loc_zw  = zeros(Int, n_components[1], n_words)
    N_loc_z   = zeros(Int, n_components[1])
    N_loc_dvz = [zeros(Int, length(X_[d][2]), n_components[1]) for d in 1:n_docs]
    N_loc_dv  = [zeros(Int, length(X_[d][2])) for d in 1:n_docs]

    N_gl_zw = zeros(Int, n_components[2], n_words)
    N_gl_z  = zeros(Int, n_components[2])
    N_gl_dz = zeros(Int, n_docs, n_components[2])
    N_gl_d  = zeros(Int, n_docs)

    srand(0)
    @inbounds for iter in 1:max_iter
        @inbounds for d in 1:n_docs
            @inbounds for (s, (N_ds, X)) in enumerate(X_[d][1])
                n = 0
                @inbounds for (w, N_dsw) in X
                    @inbounds for _ in 1:N_dsw
                        n += 1

                        # remove w_dsn’s statistics
                        if z_dsn[d][s][n] != 0
                            r = r_dsn[d][s][n]
                            z = z_dsn[d][s][n]
                            v = v_dsn[d][s][n]

                            N_dsv[d][s,v] -= 1
                            N_dvr[d][v,r] -= 1
                            N_dv[d][v]    -= 1

                            # loc
                            if r == 1
                                N_loc_zw[z,w]     -= 1
                                N_loc_z[z]        -= 1
                                N_loc_dvz[d][v,z] -= 1
                                N_loc_dv[d][v]    -= 1

                            # gl
                            elseif r == 2
                                N_gl_zw[z,w] -= 1
                                N_gl_z[z]    -= 1
                                N_gl_dz[d,z] -= 1
                                N_gl_d[d]    -= 1
                            end
                        end

                        log_p_loc_vz = zeros(T, n_components[1])
                        log_p_gl_vz  = zeros(T, n_components[2])
                        @inbounds for t in 1:T
                            v = s + t - 1

                            # loc
                            log_p_loc_vz[t,:] += log(N_dsv[d][s,v] + γ) # - log(N_ds + γ * T)
                            log_p_loc_vz[t,:] += log(N_dvr[d][v,1] + a) # - log(X_[d][2][v] + a + b)
                            for k in 1:n_components[1]
                                log_p_loc_vz[t,k] += log(N_loc_zw[k,w] + β) - log(N_loc_z[k] + β * n_words)
                                log_p_loc_vz[t,k] += log(N_loc_dvz[d][v,k] + α_loc) - log(N_loc_dv[d][v] + α_loc * n_components[1])
                            end

                            # gl
                            log_p_gl_vz[t,:] += log(N_dsv[d][s,v] + γ) # - log(N_ds + γ * T)
                            log_p_gl_vz[t,:] += log(N_dvr[d][v,2] + b) # - log(X_[d][2][v] + a + b)
                            for k in 1:n_components[2]
                                log_p_gl_vz[t,k] += log(N_gl_zw[k,w] + β) - log(N_gl_z[k] + β * n_words)
                                log_p_gl_vz[t,k] += log(N_gl_dz[d,k] + α_gl) - log(N_gl_d[d] + α_gl * n_components[2])
                            end
                        end

                        # sample v_dsn, r_dsn, and z_dsn
                        log_p_vz = collect(Iterators.flatten(cat(2, log_p_loc_vz, log_p_gl_vz)))
                        p_vz     = exp.(log_p_vz - maximum(log_p_vz))

                        ind_dsn        = sample(1:length(p_vz), WeightVec(p_vz / sum(p_vz)))
                        v_dsn[d][s][n] = s + rem(ind_dsn - 1, T)

                        if ind_dsn <= T * n_components[1]
                            r_dsn[d][s][n] = 1
                            z_dsn[d][s][n] = div(ind_dsn - 1, T) + 1
                        else
                            r_dsn[d][s][n] = 2
                            z_dsn[d][s][n] = div(ind_dsn - T * n_components[1] - 1, T) + 1
                        end

                        # add w_dsn’s statistics
                        r = r_dsn[d][s][n]
                        z = z_dsn[d][s][n]
                        v = v_dsn[d][s][n]

                        N_dsv[d][s,v] += 1
                        N_dvr[d][v,r] += 1
                        N_dv[d][v]    += 1

                        # loc
                        if r == 1
                            N_loc_zw[z,w]     += 1
                            N_loc_z[z]        += 1
                            N_loc_dvz[d][v,z] += 1
                            N_loc_dv[d][v]    += 1

                        # gl
                        elseif r == 2
                            N_gl_zw[z,w] += 1
                            N_gl_z[z]    += 1
                            N_gl_dz[d,z] += 1
                            N_gl_d[d]    += 1
                        end
                    end
                end
            end
        end
    end

    posteriori_estimation(corpus, n_components, T, N_dsv, N_dvr, N_loc_zw, N_loc_dvz, N_gl_zw, N_gl_dz)
end

function posteriori_estimation(corpus::Vector, n_components::Tuple, T::Int, N_dsv::Array, N_dvr::Array, N_loc_zw::Array, N_loc_dvz::Array, N_gl_zw::Array, N_gl_dz::Array)
    # θ_loc_dsz = Vector{Array{Float64,2}}()
    # θ_gl_dsz  = Vector{Array{Float64,2}}()
    # θ_loc_dvz = Vector{Any}()
    # θ_gl_dz   = (N_gl_dz + α_gl) ./ (sum(N_gl_dz, 2) + α_gl * n_components[2])
    # π_loc_dv  = Vector{Any}()
    # π_gl_dv   = Vector{Any}()
    # ψ_dsv     = Vector{Any}()
    # for d in 1:size(N_dsv, 1)
    #     push!(θ_loc_dvz, (N_loc_dvz[d] + α_loc) ./ (sum(N_loc_dvz[d], 2) + α_loc * n_components[1]))
    #     push!(π_loc_dv, (N_dvr[d][:,1] + a) ./ (sum(N_dvr[d], 2) + a + b))
    #     push!(π_gl_dv, (N_dvr[d][:,2] + b) ./ (sum(N_dvr[d], 2) + a + b))
    #
    #     push!(ψ_dsv, zeros(size(N_dsv[d])))
    #     S = size(N_dsv[d], 1)
    #     push!(θ_loc_dsz, zeros(S, n_components[1]))
    #     push!(θ_gl_dsz, zeros(S, n_components[2]))
    #     for s in 1:S
    #         for t in 1:T
    #             v = s + t - 1
    #             ψ_dsv[d][s,v] = (N_dsv[d][s,v] + γ) / (sum(N_dsv[d][s,:]) + γ * T)
    #
    #             θ_loc_dsz[d][s,:] += ψ_dsv[d][s,v] * π_loc_dv[d][v] .* θ_loc_dvz[d][v,:]
    #             θ_gl_dsz[d][s,:]  += ψ_dsv[d][s,v] * π_gl_dv[d][v] .* θ_gl_dz[d,:]
    #         end
    #     end
    # end

    n_words = length(corpus)
    ϕ_loc   = (N_loc_zw + β) ./ (sum(N_loc_zw, 2) + β * n_words)
    ϕ_gl    = (N_gl_zw + β) ./ (sum(N_gl_zw, 2) + β * n_words)
    ϕ_zw    = cat(1, Φ_loc, Φ_gl)

    global params = returns(ϕ_zw, corpus, n_components)
end

end # module
