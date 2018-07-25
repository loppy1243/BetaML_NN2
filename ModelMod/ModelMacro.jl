@reexport module ModelMacro
export @model

import Flux
import ..ModelMod

using Loppy.Util: fcat

is_pair_expr(x::Expr) = x.head == :call && x.args[1] == :(=>)
is_pair_expr(x) = false

is_kw_expr(x::Expr) = x.head == :kw
is_kw_expr(x) = false

is_param_expr(x::Expr) = x.head == :parameters
is_param_expr(x) = false

param_sym(x::Symbol) = esc(x)
param_sym(x::Expr) = 
    if is_pair_expr(x)
        :(ModelMod.Placeholder(ModelMod.PlaceholderName($(param_sym(x.args[3]))),
                               $(param_sym(x.args[2]))))
    else
        param_sym(x.args[1])
    end

param_name(x::Symbol) = string(x)
param_name(x::Expr) =
    if is_pair_expr(x)
        param_name(x.args[2])
    else
        param_name(x.args[1])
    end

param_esc(x::Symbol) = esc(x)
param_esc(x::Expr) =
    if is_pair_expr(x)
        Expr(:(::), gensym(), :(Pair{$(param_ty(x.args[2])), $(param_ty(x.args[3]))}))
    elseif is_kw_expr(x)
        Expr(:kw, param_esc(x.args[1]), esc(x.args[2]))
    else
        esc(x)
    end

param_pairs(x::Symbol) = nothing
param_pairs(x::Expr) =
    if is_pair_expr(x)
        (param_sym(x.args[2]), param_sym(x.args[3]))
    else
        nothing
    end

is_ty_expr(x::Symbol) = false
is_ty_expr(x::Expr) = x.head == :(::)

param_ty(x::Symbol) = :(<:Any)
param_ty(x::Expr) =
    if is_kw_expr(x)
        param_ty(x.args[1])
    elseif is_ty_expr(x)
        esc(x.args[2])
    else
        :(<:Any)
    end

getname(x::Symbol) = esc(x)
getname(x::Expr) = x.head == :(<:) ? esc(x.args[1]) : esc(x)

getsupertype(x::Symbol) = :(ModelMod.ModelType)
getsupertype(x::Expr) = x.head == :(<:) ? esc(x.args[2]) : :(ModelMod.Modeltype)

macro model(expr::Expr)
    @assert expr.head == :function || expr.head == :(=) && expr.args[1] isa Expr #=
         =# && expr.args[1].head == :call

    name, sup_ty = expr.args[1].args[1] |> fcat(getname, getsupertype)
    hyparams = expr.args[1].args[2:end]
    body = expr.args[2]

    kwparams = if is_param_expr(hyparams[1])
        x = hyparams[1].args
        @assert all(is_kw_expr, x)

        hyparams = hyparams[2:end]

        x
    else
        []
    end

    hyparam_escs  = [Expr(:parameters, map(param_esc, kwparams)...); map(param_esc,   hyparams)]
    hyparam_pairs = [map(param_pairs, kwparams)                    ; map(param_pairs, hyparams)]
    hyparam_syms  = [map(param_sym,   kwparams)                    ; map(param_sym,   hyparams)]
    hyparam_names = [map(param_name,  kwparams)                    ; map(param_name,  hyparams)]

    hyparam_pairs = filter(x -> x != nothing, hyparam_pairs)
    pair_arg_syms = filter(x -> x.head == :(::), hyparam_escs) |> #=
                 =# x -> map(y -> y.args[1], x)
    @assert length(hyparam_pairs) == length(pair_arg_syms)

    exprs = map(hyparam_pairs, pair_arg_syms) do ps, arg
        val_sym, name_sym = ps

        quote
            $val_sym = first($arg)
            $name_sym = last($arg)
        end
    end

    quote
        abstract type $name <: $sup_ty end

        function (::Base.Type{T})($(hyparam_escs[2:end]...); id::Union{Integer, Void}=nothing,
                                  $(hyparam_escs[1].args...)) where T<:$name
            $(exprs...)
            func = $(esc(body))
            params = Flux.params(func)
            hyparams = Dict(zip(($hyparam_names...), ($(hyparam_syms...),)))
            real_id = id === nothing ? hash(string(T)) + hash(hyparams) : convert(UInt, id)

            ModelMod.Model{$name, typeof(func)}(real_id, func, params, hyparams)
        end
    end
end

end # module ModelMacro
