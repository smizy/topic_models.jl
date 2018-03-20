module topic_models

module Bayesian_mixtures

# models
include("./LDA/LDA.jl")
include("./HDP/HDP.jl")

# utils
include("./corpora.jl")

end # module
