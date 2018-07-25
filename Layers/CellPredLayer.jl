@reexport module CellPredLayer
export CellPred

using Flux
using BetaML.Data: GRIDSIZE
using ..ConcatLayer

cellpred(dists) = mapslices(dists, 1:2) do dist
    ind2sub(dist, indmax(dist)) |> collect
end |> x -> squeeze(x, 2)

CellPred() = Chain(Conv((3, 3), 1=>1, pad=(1, 1)),
                   x -> reshape(x, prod(GRIDSIZE), :),
                   softmax,
                   x -> reshape(x, GRIDSIZE..., :),
                   @Concat(identity, cellpred))

end # module CellPredLayer
