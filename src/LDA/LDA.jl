module LDA

struct returns
    Φ_::Array{Float64,2}
    corpus::Vector{String}
    n_components::Int
end

using StatsBase

const α = 0.5
const β = 0.1

function fit(X_::Array, corpus::Vector; max_iter::Int=100, n_components::Int=10)
    @assert max_iter > 0 && n_components > 0

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

    return [params.corpus[v] for v in sortperm(params.Φ_[topicid,:], rev=true)[1:topn]]
end

# estimate model parameters with collapsed Gibbs sampling
function cgs(X_::Array, corpus::Vector, max_iter::Int, n_components::Int)
    n_words = length(corpus)

    z_dn = [zeros(Int, N_d) for (N_d, X) in X_]
    N_dk = zeros(Int, length(X_), n_components)
    N_kv = zeros(Int, n_components, n_words)
    N_k  = zeros(Int, n_components)

    srand(0)
    @inbounds for iter in 1:max_iter
        @inbounds for (d, (N_d, X)) in enumerate(X_)
            n = 0
            @inbounds for (v, N_dv) in X
                @inbounds for _ in 1:N_dv
                    n += 1

                    # remove w_dv’s statistics
                    if z_dn[d][n] != 0
                        k = z_dn[d][n]
                        N_dk[d,k] -= 1
                        N_kv[k,v] -= 1
                        N_k[k]    -= 1
                    end

                    log_p_k = zeros(n_components)
                    @inbounds for k in 1:n_components
                        log_p_k[k]  = log(N_dk[d,k] + α)
                        log_p_k[k] += log(N_kv[k,v] + β) - log(N_k[k] + β * n_words)
                    end

                    # sample z_dn after normalizing
                    p_k = exp.(log_p_k - maximum(log_p_k))
                    z_dn[d][n] = sample(1:n_components, WeightVec(p_k / sum(p_k)))

                    # add w_dv’s statistics
                    k = z_dn[d][n]
                    N_dk[d,k] += 1
                    N_kv[k,v] += 1
                    N_k[k]    += 1
                end
            end
        end
    end

    posteriori_estimation(corpus, n_components, N_dk, N_kv)
end

function posteriori_estimation(corpus::Vector, n_components::Int, N_dk::Array, N_kv::Array)
    # θ_dk = (N_dk + α) ./ (sum(N_dk, 2) + α * n_components)
    ϕ_kv = (N_kv + β) ./ (sum(N_kv, 2) + β * length(corpus))

    global params = returns(ϕ_kv, corpus, n_components)
end

end # module
