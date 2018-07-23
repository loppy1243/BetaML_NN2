module ModelMod
export Model, ModelType, predict, loss

import Flux

using Reexport: @reexport
using Flux.Tracker: istracked, data

abstract type ModelType end

struct Model{T<:ModelType, F}
    func::F
    hyparams::Dict{String}

    Model{T, F}(func::F, hyparams::Dict) where {T<:ModelType, F} = new(func, hyparams)
    Model{T, F}(func::F, hyparams...) where {T<:ModelType, F} = new(func, Dict(hyparams))
end

Flux.params(m::Model) = Flux.params(m.func)
hyparams(m::Model) = m.hyparams
hyparams(m::Model, itr) = [m.hyparams[k] for k in itr]
hyparams(m::Model, k::String) = m.hyparams[k]
hyparams(m::Model, ks::Vararg{String}) = hyparams(m, ks)

predict(m::Model) = (args...,) -> predict(m, args...)
function predict(m::Model, args...)
    x = m(args...)
    istracked(x) ? data(x) : map(data, x)
end

loss(m::Model) = (args...,) -> loss(m, args...)

include("ModelMod/ModelMacro.jl")

end # module ModelMod
