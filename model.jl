import Flux
using Flux.Tracker: istracked, data

abstract type ModelType

struct Model{T<:ModelType}
    params::Vector{Any}
    hyparams::Dict{String}

    Model{<:ModelType}(params, hyparams::Dict) = new(params, hyparams)
    Model{<:ModelType}(params, hyparams...) = new(params, Dict(hyparams))
end

predict(m::Model) = (args...,) -> predict(m, args...)
function predict(m::Model, args...)
    x = m(args...)
    istracked(x) ? data(x) : map(data, x)
end

loss(m::Model) = (args...,) -> loss(m, args...)

param_sym(x::Symbol) = $(esc(x))
param_sym(x::Expr) = 
    x.head == :call && x.args[1] == :(=>) ? param_sym(x.args[2]) : param_sym(x.args[1])

param_name(x::Symbol) = string(x)
param_name(x::Expr) =
    x.head == :call && x.args[1] == :(=>) ? x.args[3] : param_name(x.args[1])

param_esc(x::Symbol) = esc(x)
param_esc(x::Expr) =
    x.head == :call && x.args[1] == :(=>) ? esc(x.args[2]) : esc(x)

macro model(expr::Expr)
    @assert expr.head == :function || expr.head == := && expr.args[1] isa Expr #=
         =# && expr.args[1].head == :call

    name = expr.args[1].args[1]
    hyparams = expr.args[1].args[2:end]
    body = expr.args[2]

    hyparams_esc = map(param_esc, hyparams)
    hyparam_syms = map(param_sym, hyparams)
    hyparam_names = map(param_name, hyparams)
    name_esc = esc(name)

    quote
        abstract type $name_esc <: ModelType end
        let model=$(esc(body))
            function Model{$name_esc}($(hyparams_esc...))
                params = Flux.params(model)
                hyparams = Dict(zip($hyparam_names, ($(hyparam_syms...),)))

                Model{$name_esc}(params, hyparams)
            end

            (::Model{$name_esc})(args...) = model(args...)
        end
    end
end
