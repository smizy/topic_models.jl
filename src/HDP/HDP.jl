module HDP

struct returns
    Φ_::Array{Float64,2}
    corpus::Vector{String}
    n_components::Int
end

using StatsBase

const α = 0.5
const β = 0.1
const γ = 0.5

function fit(X_::Array, corpus::Vector; max_iter::Int=100)
    @assert max_iter > 0

    cgs(X_, corpus, max_iter)
end

# get parameters for this estimator
function get_params()
    @assert isdefined(params, :corpus)

    return params
end

# return the most probable topn words in topic topicid
function show_topic(topicid::Int; topn::Int=10)
    @assert isdefined(params, :corpus)
    @assert 1 <= topicid <= params.n_components
    @assert topn <= length(params.corpus)

    return [params.corpus[v] for v in sortperm(params.Φ_[topicid,:], rev=true)[1:topn]]
end

# estimate model parameters with collapsed Gibbs sampling
function cgs(X_::Array, corpus::Vector, max_iter::Int)
    n_words = length(corpus)

    t_dn  = [zeros(Int, N_d) for (N_d, X) in X_]
    z_dℓ  = [zeros(Int, 0) for _ in X_]
    N_dℓv = [zeros(Int, 0, n_words) for _ in X_]
    N_dℓ  = [zeros(Int, 0) for _ in X_]
    N_kv  = zeros(Int, 0, n_words)
    N_k   = zeros(Int, 0)
    M_k   = zeros(Int, 0)
    M     = 0
    n_components = 0
    n_tables     = zeros(Int, length(X_))

    srand(0)
    @inbounds for iter in 1:max_iter
        @inbounds for (d, (N_d, X)) in enumerate(X_)

            # sampling t_dn
            n = 0
            @inbounds for (v, N_dv) in X
                @inbounds for _ in 1:N_dv
                    n += 1

                    # remove w_dv’s statistics
                    if t_dn[d][n] != 0
                        ℓ = t_dn[d][n]
                        k = z_dℓ[d][ℓ]
                        N_dℓv[d][ℓ,v] -= 1
                        N_dℓ[d][ℓ]    -= 1
                        N_kv[k,v]     -= 1
                        N_k[k]        -= 1

                        # if any component is empty, remove it and decrease n_tables
                        if N_dℓ[d][ℓ] == 0
                            n_tables[d] -= 1
                            M_k[k]      -= 1
                            M           -= 1
                            deleteat!(z_dℓ[d], ℓ)
                            deleteat!(N_dℓ[d], ℓ)
                            N_dℓv[d] = N_dℓv[d][setdiff(1:end, ℓ),:]
                            @inbounds for (tmp_n, tmp_t) in enumerate(t_dn[d])
                                if tmp_t > ℓ
                                    t_dn[d][tmp_n] -= 1
                                end
                            end

                            # if any component is empty, remove it and decrease n_components
                            if M_k[k] == 0
                                n_components -= 1
                                deleteat!(M_k, k)
                                deleteat!(N_k, k)
                                N_kv = N_kv[setdiff(1:end, k),:]

                                @inbounds for tmp_d in 1:length(X_)
                                    @inbounds for (tmp_ℓ, tmp_z) in enumerate(z_dℓ[tmp_d])
                                        if tmp_z > k
                                            z_dℓ[tmp_d][tmp_ℓ] -= 1
                                        end
                                    end
                                end
                            end
                        end
                    end

                    L       = n_tables[d] + n_components + 1
                    log_p_ℓ = zeros(L)

                    # existing components
                    @inbounds for ℓ in 1:n_tables[d]
                        log_p_ℓ[ℓ]  = log(N_dℓ[d][ℓ])
                        log_p_ℓ[ℓ] += log(N_kv[z_dℓ[d][ℓ], v] + β) - log(N_k[z_dℓ[d][ℓ]] + β * n_words)
                    end

                    # new components
                    @inbounds for k in 1:n_components
                        ℓ = n_tables[d] + k
                        log_p_ℓ[ℓ]  = log(α)
                        log_p_ℓ[ℓ] += log(M_k[k]) - log(M + γ)
                        log_p_ℓ[ℓ] += log(N_kv[k,v] + β) - log(N_k[k] + β * n_words)
                    end

                    log_p_ℓ[L]  = log(α)
                    log_p_ℓ[L] += log(γ) - log(M + γ)
                    log_p_ℓ[L] += log(β) - log(β * n_words)

                    # sample t_dn after normalizing
                    p_ℓ = exp.(log_p_ℓ - maximum(log_p_ℓ))
                    ℓ = sample(1:L, WeightVec(p_ℓ / sum(p_ℓ)))

                    if ℓ <= n_tables[d]
                        t_dn[d][n] = ℓ
                    else
                        t_dn[d][n] = n_tables[d] + 1
                        push!(z_dℓ[d], 0)
                        push!(N_dℓ[d], 0)
                        N_dℓv[d] = cat(1, N_dℓv[d], zeros(Int, 1, n_words))

                        if ℓ - n_tables[d] <= n_components
                            z_dℓ[d][t_dn[d][n]] = ℓ - n_tables[d]
                        else
                            z_dℓ[d][t_dn[d][n]] = n_components + 1
                            n_components += 1
                            push!(M_k, 0)
                            push!(N_k, 0)
                            N_kv = cat(1, N_kv, zeros(Int, 1, n_words))
                        end

                        n_tables[d] += 1
                        M_k[z_dℓ[d][t_dn[d][n]]] += 1
                        M                        += 1
                    end

                    # add w_dv’s statistics
                    ℓ = t_dn[d][n]
                    k = z_dℓ[d][ℓ]
                    N_dℓv[d][ℓ,v] += 1
                    N_dℓ[d][ℓ]    += 1
                    N_kv[k,v]     += 1
                    N_k[k]        += 1
                end
            end

            # sampling z_dℓ
            @inbounds for ℓ in 1:n_tables[d]

                # remove w_dℓ’s statistics
                if z_dℓ[d][ℓ] != 0
                    k = z_dℓ[d][ℓ]
                    M_k[k] -= 1
                    @inbounds for (v, N_dv) in X
                        N_kv[k,v] -= N_dℓv[d][ℓ,v]
                        N_k[k]    -= N_dℓv[d][ℓ,v]
                    end

                    # if any component is empty, remove it and decrease n_components
                    if M_k[k] == 0
                        n_components -= 1
                        deleteat!(M_k, k)
                        deleteat!(N_k, k)
                        N_kv = N_kv[setdiff(1:end, k),:]

                        @inbounds for tmp_d in 1:length(X_)
                            @inbounds for (tmp_ℓ, tmp_z) in enumerate(z_dℓ[tmp_d])
                                if tmp_z > k
                                    z_dℓ[tmp_d][tmp_ℓ] -= 1
                                end
                            end
                        end
                    end
                end

                K = n_components + 1
                log_p_k = zeros(K)

                # existing components
                @inbounds for k in 1:n_components
                    log_p_k[k]  = log(M_k[k])
                    log_p_k[k] += lgamma(N_k[k] + β * n_words) - lgamma(N_k[k] + N_dℓ[d][ℓ] + β * n_words)
                    @inbounds for (v, N_dv) in X
                        log_p_k[k] += lgamma(N_kv[k,v] + N_dℓv[d][ℓ,v] + β) - lgamma(N_kv[k,v] + β)
                    end
                end

                # a new component
                log_p_k[K]  = log(γ)
                log_p_k[K] += lgamma(β * n_words) - lgamma(N_dℓ[d][ℓ] + β * n_words)
                @inbounds for (v, N_dv) in X
                    log_p_k[K] += lgamma(N_dℓv[d][ℓ,v] + β) - lgamma(β)
                end

                # sample z_dℓ after normalizing
                p_k = exp.(log_p_k - maximum(log_p_k))
                z_dℓ[d][ℓ] = sample(1:K, WeightVec(p_k / sum(p_k)))

                if z_dℓ[d][ℓ] == K
                    n_components += 1
                    push!(M_k, 0)
                    push!(N_k, 0)
                    N_kv = cat(1, N_kv, zeros(Int, 1, n_words))
               end

                # add w_dℓ’s statistics
                k = z_dℓ[d][ℓ]
                M_k[k] += 1
                @inbounds for (v, N_dv) in X
                    N_kv[k,v] += N_dℓv[d][ℓ,v]
                    N_k[k]    += N_dℓv[d][ℓ,v]
                end
            end
        end
    end

    N_dk = zeros(Int, length(X_), n_components)
    @inbounds for d in 1:length(X_)
        @inbounds for (ℓ, k) in enumerate(z_dℓ[d])
            N_dk[d, k] += N_dℓ[d][ℓ]
        end
    end

    posteriori_estimation(corpus, n_components, N_dk, N_kv)
end

function posteriori_estimation(corpus::Vector, n_components::Int, N_dk::Array, N_kv::Array)
    # θ_dk = N_dk ./ sum(N_dk, 2)
    ϕ_kv = (N_kv + β) ./ (sum(N_kv, 2) + β * length(corpus))

    global params = returns(ϕ_kv, corpus, n_components)
end

end # module
