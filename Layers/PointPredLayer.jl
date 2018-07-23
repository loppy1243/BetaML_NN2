@reexport module PointPredLayer
export PointPred

using Flux

using BetaML.Data
using Flux.Tracker: data
using Lazy: @>

const NODES = prod(2(GRIDSIZE.-1).+1)
const SCALE = (XYMAX - XYMIN)./2GRIDSIZE

struct PointPred{T, F<:Function}
    denselayer::T
    activ::F
    cutgrad::Bool
end
function PointPred(activ, N; cutgrad=false)
    l = Chain(Dense(NODES, N, activ), Dense(N, 2))

    PointPred{typeof(l), typeof(activ)}(l, activ, cutgrad)
end

Flux.params(p::PointPred) = Flux.params(p.denselayer)

(p::PointPred)(x) = p.cutgrad ? _pointpred(p, data(x[1])) : _pointpred(p, x[x])
_pointpred(p::PointPred, x) = @> x begin
    recenter(x[2])
    p.activ.()
    vec
    p.denselayer
    (*).(SCALE)
end

function recenter(dists, cells)
    ret = zeros(eltype(dists), (2GRIDSIZE .+ 1)..., size(dists, 3))

    for k = 1:GRIDSIZE[1], l = 1:GRIDSIZE[2], i in indices(dists, 3)
        ixs = [k, l] - cells[:, i] + GRIDSIZE .+ 1

        ret[ixs..., i] = dists[k, l, i]
    end

    Tracker.collect(ret)
end

end # module PointPredLayer
