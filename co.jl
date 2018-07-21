module COModel

using Flux

using BetaML_Data
using Model

regularize(x) = reshape(x/MAX_E, GRIDSIZE, 1, :)

@model CO(activ, N, ϵ, λ) =
    Chain(regularize, CellLayer(), @Concat(identity..., PointLayer(activ, N)))

#function Model.loss(m::Model{CO}, events, points)
#    pred_dists, pred_cells, pred_rel_points = m(events)
#    cells = mapslices(pointcell, points, 1)
#
#    sum(indices(points, 2)) do i
#        pred_prob = pred_dists[pred_cells[:, i]..., i]
#
#        -log(ϵ + max(0, pred_prob)) + log(1+ϵ - min(1, pred_prob)) #=
#     =# - sum(
#    end
#end

end # module COModel
