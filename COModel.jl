@reexport module COModel
export CO

using Flux
using Plots

import ..ModelMod: loss, predict, train

using BetaML.Data
using ..ModelMod, ..Layers
using Loppy.Util: batch

regularize(x) = reshape(x/MAX_E, GRIDSIZE..., 1, :)

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
     =# + λ*sum((pred_point - point).^2)
    end
end

function make_info_f(f, numbatches, model, events, points)
    batch = 1
    () -> (relay_info(numbatches, batch, model, events, points); batch += 1)
end

function relay_info(numbatches, batch, model, events, points)
    numevents = size(events, 3)

    (pred_dists, _, pred_points) = predict(model, events)
    i = rand(indices(events, 3))

    point = points[:, i]
    pred_dist = pred_dists[:, :, i]
    pred_point = pred_points[:, i]

    lossval = loss(model, events, points) |> data
    println("$(lpad(batch, ndigits(numbatches), 0))/$numbatches: ",
            "Event $(lpad(i, numevents, 0)) | Loss = $(signif(lossval, 4))")
    spy(pred_dist)
    plotpoint!(point, color=:green)
    plotpoint!(pred_point, color=:red)
end

const BATCHSIZE=1000

function ModelMod.train(file, model::Model{<:CO}, events, points; load=true)
    load ? load!(file, model) : save(file, model)
    batches = zip(batch(events, BATCHSIZE), batch(points, BATCHSIZE)) |> collect

    save_t = Flux.throttle(12) do; save(file, model) end
    relay_info_t = begin
        f = make_info_f(relay_info, length(batches), model, events, points)
        Flux.throttle(f, 3)
    end

    shuffle!(batches)

    opt = hyparams(model, "opt")(params(model), hyparams(model, "η"))
    try
        Flux.Optimise.train!(loss(model), batches, opt, cb=[save_t, relay_info_t])
    catch ex
        ex isa InterruptException ? interrupt() : rethrow()
    finally
        save(file, model)
    end
end

end # module COModel
