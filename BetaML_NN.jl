module BetaML_NN

using Reexport: @reexport

include("ModelMod/main.jl")
include("Layers/main.jl"); export Layers
include("COModel.jl"); export COModel
include("Tests.jl")

end # module BetaML_NN
