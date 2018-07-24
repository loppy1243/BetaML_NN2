@reexport module ModelMod
export Model, ModelType, predict, loss, train, save, load

import Flux
import JLD2

using Reexport: @reexport
using Flux.Tracker: istracked, data

abstract type ModelType end

struct Model{T<:ModelType, F}
    id::UInt
    func::F
    params::Vector
    hyparams::Dict{String}

    Model{T, F}(id, func::F, params, hyparams::Dict) where {T<:ModelType, F} =
        new(id, func, params, hyparams)
    Model{T, F}(id, func::F, params, hyparams...) where {T<:ModelType, F} =
        new(id, func, params, Dict(hyparams))
end
(m::Model)(args...) = m.func(args...)

struct PlaceholderName; val::String end
struct Placeholder{T}
    name::PlaceholderName
    val::T
end
Placeholder(name, x) = Placeholder{typeof(x)}(PlaceholderName(name), x)
hash(x::Placeholder, h) = hash(x.name, h)

modelname(m::Model{T}) where T<:ModelType = string(T)
modelid(m::Model) = m.id

Flux.params(m::Model) = m.params

hyparams(m::Model) = map(x -> x[1]=>_proc_hyparam(x[2]), m.hyparams)
hyparams(m::Model, itr) = [_proc_hyparam(m.hyparams[k]) for k in itr]
hyparams(m::Model, k::String) = _proc_hyparam(m.hyparams[k])
hyparams(m::Model, ks::Vararg{String}) = hyparams(m, ks)

_proc_hyparam(x::Placeholder) = x.val
_proc_hyparam(x) = x

predict(m::Model) = (args...,) -> predict(m, args...)
function predict(m::Model, args...)
    x = m(args...)
    istracked(x) ? data(x) : map(data, x)
end

loss(m::Model) = (args...,) -> loss(m, args...)

train(args...) = throw(MethodError(train, (args...)))

save(m::Model, file) = JLD2.jldopen(file, "a") do io
    mname = modelname(m)
    grp = JLD2.Group(haskey(io, mname) ? io[mname] : JLD2.Group(io, mname),
                     string(modelid(m)))
    grp["params"] = data(m.params)
    grp["hyparams"] = map(m.hyparams) do x
        x[1] => (x[2] isa Placeholder ? x[2].name : x[2])
    end
end

load!(m::Model, file) = JLD2.jldopen(file, "r") do io
    grp = io[modelname(m)][string(modelid(m))]

    for (p, p_dat) in zip(m.params, grp["params"])
        p.tracker.data = p_dat
    end

    for (k, v) in grp["hyparams"]
        v isa Placeholder && continue
        m.hyparams[k] = v
    end

    m
end

include("ModelMod/ModelMacro.jl")

end # module ModelMod
