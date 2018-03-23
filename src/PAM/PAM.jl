module PAM

struct returns
    Φ_::Array{Float64,2}
    corpus::Vector{String}
    n_components::Int
end

using StatsBase

const α = 0.5
const β = 0.1
const γ = 0.1

function fit(X_::Array, corpus::Vector; max_iter::Int=100, n_components::Tuple=(10,30))
    @assert max_iter > 0 && n_components[1] > 0 && n_components[2] > 0

    cgs(X_, corpus, max_iter, n_components)
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

    return [params.corpus[v] for v in sortperm(params.Φ_[topicid, :], rev=true)[1:topn]]
end

# estimate model parameters with collapsed Gibbs sampling
function cgs(X_::Array, corpus::Vector, max_iter::Int, n_components::Tuple)
    n_words = length(corpus)

    y_dn  = [zeros(Int, N_d) for (N_d, X) in X_]
    z_dn  = [zeros(Int, N_d) for (N_d, X) in X_]
    N_dsk = zeros(Int, length(X_), n_components[1], n_components[2])
    N_ds  = zeros(Int, length(X_), n_components[1])
    N_kv  = zeros(Int, n_components[2], n_words)
    N_k   = zeros(Int, n_components[2])

    srand(0)
    @inbounds for iter in 1:max_iter
        @inbounds for (d, (N_d, X)) in enumerate(X_)
            n = 0
            @inbounds for (v, N_dv) in X
                @inbounds for _ in 1:N_dv
                    n += 1

                    # remove w_dv’s statistics
                    if z_dn[d][n] != 0
                        s = y_dn[d][n]
                        k = z_dn[d][n]
                        N_dsk[d, s, k] -= 1
                        N_ds[d, s]     -= 1
                        N_kv[k, v]     -= 1
                        N_k[k]         -= 1
                    end

                    log_p_yz = zeros(n_components)
                    @inbounds for s in 1:n_components[1]
                        @inbounds for k in 1:n_components[2]
                            log_p_yz[s, k]  = log(N_ds[d, s] + α) - log(N_d + α * n_components[1])
                            log_p_yz[s, k] += log(N_dsk[d, s, k] + γ) - log(N_ds[d, s] + γ * n_components[2])
                            log_p_yz[s, k] -= log(N_kv[k, v] + β) - log(N_k[k] + β * n_words)
                        end
                    end

                    # sample y_dn and z_dn after normalizing
                    log_p_yz = collect(Iterators.flatten(log_p_yz))
                    p_yz     = exp.(log_p_yz - maximum(log_p_yz))

                    ind_dn     = sample(1:length(p_yz), WeightVec(p_yz / sum(p_yz)))
                    y_dn[d][n] = rem(ind_dn - 1, n_components[1]) + 1
                    z_dn[d][n] = div(ind_dn - 1, n_components[1]) + 1

                    # add w_dv’s statistics
                    s = y_dn[d][n]
                    k = z_dn[d][n]
                    N_dsk[d, s, k] += 1
                    N_ds[d, s]     += 1
                    N_kv[k, v]     += 1
                    N_k[k]         += 1
                end
            end
        end
    end

    posteriori_estimation(corpus, n_components, N_dsk, N_kv)
end

function posteriori_estimation(corpus::Vector, n_components::Tuple, N_dsk::Array, N_kv::Array)
    # θ_ds  = (sum(N_dsk, 3) + α) ./ (sum(N_dsk, (2, 3)) + α * n_components[1])
    # ψ_dsk = (N_dsk + γ) ./ (sum(N_dsk, 3) + γ * n_components[2])

    ϕ_kv = (N_kv + β) ./ (sum(N_kv, 2) + β * length(corpus))

    global params = returns(ϕ_kv, corpus, n_components[2])
end

end # module
