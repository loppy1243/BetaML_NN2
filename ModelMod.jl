module ModelMod
export Model, ModelType, predict, loss

import Flux
import JLD2

using Reexport: @reexport
using Flux.Tracker: istracked, data

abstract type ModelType end

struct Model{T<:ModelType, F}
    name::String
    func::F
    hyparams::Dict{String}

    Model{T, F}(id, func::F, hyparams::Dict) where {T<:ModelType, F} = new(id, func, hyparams)
    Model{T, F}(id, func::F, hyparams...) where {T<:ModelType, F} = new(id, func, Dict(hyparams))
end

modelname(m::Model) = m.name

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

function save(m::Model, file)
    JLD2.jldopen(file, "a") do io
        grp = JLD2.Group(io, modelname(m))
        grp["params"] = Flux.params(m)

        hyps = hyparams(m)
        hyps_grp = JLD2.Group(grp, "hyparams")
        for k in keys(hyps)
            hyps_grp[k] = hyps[k]
        end
    end
end

# TODO
function load(m::Model, file)
    JLD2.jldopen(file, "r") do io
    end
end

include("ModelMod/ModelMacro.jl")

end # module ModelMod
