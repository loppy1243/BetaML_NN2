module ModelMod
export Model, ModelType, predict, loss, @model

import Flux
using Flux.Tracker: istracked, data

abstract type ModelType end

struct Model{T<:ModelType}
    params::Vector{Any}
    hyparams::Dict{String}

    Model{T}(params, hyparams::Dict) where T = new(params, hyparams)
    Model{T}(params, hyparams...) where T = new(params, Dict(hyparams))
end

Flux.params(m::Model) = m.params
hyparams(m::Model) = m.hyparams

predict(m::Model) = (args...,) -> predict(m, args...)
function predict(m::Model, args...)
    x = m(args...)
    istracked(x) ? data(x) : map(data, x)
end

loss(m::Model) = (args...,) -> loss(m, args...)

is_pair_expr(x::Expr) = x.head == :call && x.args[1] == :(=>)
is_pair_expr(x) = false

is_kw_expr(x::Expr) = x.head == :kw
is_kw_expr(x) = false

is_param_expr(x::Expr) = x.head == :parameters
is_param_expr(x) = false

param_sym(x::Symbol) = esc(x)
param_sym(x::Expr) = 
    if is_pair_expr(x)
        param_sym(x.args[2])
    elseif is_kw_expr(x)
        param_sym(x.args[1])
    else
        param_sym(x.args[1])
    end

param_name(x::Symbol) = string(x)
param_name(x::Expr) =
    if is_pair_expr(x)
        x.args[3]
    elseif is_kw_expr(x)
        param_name(x.args[1])
    else
        param_name(x.args[1])
    end

param_esc(x::Symbol) = esc(x)
param_esc(x::Expr) =
    if is_pair_expr(x)
        esc(x.args[2])
    elseif is_kw_expr(x)
        Expr(:kw, param_esc(x.args[1]), esc(x.args[2]))
    else
        esc(x)
    end

# TODO: Fix a=>b syntax so it works correctly
macro model(expr::Expr)
    @assert expr.head == :function || expr.head == :(=) && expr.args[1] isa Expr #=
         =# && expr.args[1].head == :call

    name = expr.args[1].args[1]
    hyparams = expr.args[1].args[2:end]
    body = expr.args[2]

    kwparams = if is_param_expr(hyparams[1])
        @assert all(is_kw_expr, hyparams[1].args)
        x = hyparams[1].args
        hyparams = hyparams[2:end]
        x
    else
        []
    end

    hyparams_esc  = [Expr(:parameters, map(param_esc, kwparams)...); map(param_esc,  hyparams)]
    hyparam_syms  = [map(param_sym, kwparams)                      ; map(param_sym,  hyparams)]
    hyparam_names = [map(param_name, kwparams)                     ; map(param_name, hyparams)]
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

end # module ModelMod
