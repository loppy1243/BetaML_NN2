module Tests

import JLD2

using Base.Test
using Flux
using ..BetaML_NN

function run()
    m1 = CO(relu=>"relu", 50, 0.01, 1.0; cutgrad=true)
    m2 = CO(relu=>"relu", 50, 0.01, 1.0; cutgrad=true)
    m3 = CO(σ=>"sigmoid", 50, 0.01, 1.0; cutgrad=true)
    m4 = CO(relu=>"relu", 50, 0.01, 1.0; cutgrad=false)
    m5 = CO(relu=>"relu", 50, 0.01, 1.0)
    m6 = CO(relu=>"relu", 51, 0.01, 1.0; cutgrad=true)

    @testset "Hashes" begin
        @test modelid(m1) == modelid(m2) && m1 == m2
        @test !(m1 in (m3, m4, m5, m6))
        @test !(m3 in (m4, m5, m6))
        @test !(m6 in (m4, m5))
        @test m4 == m5
    end

    @testset "Saving/Loading" begin
        mkpath("test")
        file = "test/models.jld2"
        rm(file, force=true)

        save(file, m1)
        save(file, m3)

        @test_throws KeyError load!(file, m2) 

        rep = Dict("relu"=>relu, "sigmoid"=>σ)
        load!(file, m2, "relu"=>relu)
        load!(file, m5, rep, id=modelid(m3))

        @test m1 == m2 && isequal(hyparams(m1), hyparams(m2))
        @test m5 == m3 && isequal(hyparams(m5), hyparams(m3))
        @test m2 != m3
    end
end

end
