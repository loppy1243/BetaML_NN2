module ModelMod
export Model, ModelType, predict, loss

using Reexport: @reexport
using Flux.Tracker: istracked, data
using Loppy.Util: includeall

abstract type ModelType end

struct Model{T<:ModelType, F}
    func::F
    hyparams::Dict{String}

    Model{T, F}(func::F, hyparams::Dict) where {T<:ModelType, F} = new(func, hyparams)
    Model{T, F}(func::F, hyparams...) where {T<:ModelType, F} = new(func, Dict(hyparams))
end

Flux.params(m::Model) = Flux.params(m.func)
hyparams(m::Model) = m.hyparams

predict(m::Model) = (args...,) -> predict(m, args...)
function predict(m::Model, args...)
    x = m(args...)
    istracked(x) ? data(x) : map(data, x)
end

loss(m::Model) = (args...,) -> loss(m, args...)

includeall("ModelMod")

end # module ModelMod
