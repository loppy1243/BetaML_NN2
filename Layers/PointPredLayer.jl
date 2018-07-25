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

(p::PointPred)(x) = p.cutgrad ? _pointpred(p, data(x[1]), data(x[2])) : _pointpred(p, x[1], x[2])
_pointpred(p::PointPred, x, y) = @> x begin
    recenter(y)
    p.activ.()
    reshape(NODES, :)
    p.denselayer
    (*).(SCALE)
end

function recenter(dists, cells)
    ret = zeros(eltype(dists), (2(GRIDSIZE.-1) .+ 1)..., size(dists, 3))

    for k = 1:GRIDSIZE[1], l = 1:GRIDSIZE[2], i in indices(dists, 3)
        ixs = [k, l] - cells[:, i] + GRIDSIZE

        ret[ixs..., i] = dists[k, l, i]
    end

    Tracker.collect(ret)
end

end # module PointPredLayer
