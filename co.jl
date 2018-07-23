module COModel

using Flux

using BetaML_Data
using BetaML_NN: ModelMod, Layers

regularize(x) = reshape(x/MAX_E, GRIDSIZE, 1, :)

@model CO(activ, N, ϵ, λ; cutgrad) =
    Chain(regularize, CellLayer(), @Concat(identity..., PointLayer(activ, N)))

function ModelMod.loss(m::Model{CO}, events, points)
    goodlog(x) = -log(ϵ + max(0, x))
    badlog(x) = -log(1+ϵ - min(1, x))

    pred_dists, pred_cells, pred_rel_points = m(events)
    cells = mapslices(pointcell, points, 1)

    sum(indices(points, 2)) do i
        point = points[:, i]
        pred_dist = pred_dists[:, :, i]
        pred_cell = pred_cells[:, i]
        pred_prob = pred_dist[pred_cell...]
        pred_point = pred_rel_points[:, i] + cellpoint(pred_cell)

        goodlog(pred_prob) - badlog(pred_prob) + sum(badlog, pred_dist) #=
     =# + λ*(pred_point - point).^2
    end
end

end # module COModel
