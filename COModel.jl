@reexport module COModel
export CO

using Flux

import ..ModelMod: loss, predict, train

using BetaML.Data
using ..ModelMod, ..Layers
using Loppy.Util: batch

regularize(x) = reshape(x/MAX_E, GRIDSIZE, 1, :)

@model CO(activ=>activ_name, opt=>opt_name, η, N, ϵ, λ; cutgrad=false) =
    Chain(regularize, CellPred(), @Concat(identity..., PointPred(activ, N, cutgrad=cutgrad)))

function loss(m::Model{<:CO}, events, points)
    ϵ, λ = hyparams(m, "ϵ", "λ")

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

const BATCHSIZE=1000

function ModelMod.train(file, model::Model{<:CO}, events, points; load=true)
    load && load!(file, model)
    batches = zip(batch(events, BATCHSIZE), batch(points, BATCHSIZE)) |> collect

    shuffle!(batches)

    opt = hyparams(model, "opt")(params(model))
    Flux.Optimise.train!(loss(model), batches, opt)
end

end # module COModel
