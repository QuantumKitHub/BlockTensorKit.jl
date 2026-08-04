using SafeTestsets

# check if user supplied args
pat = r"(?:--group=)(\w+)"
arg_id = findfirst(contains(pat), ARGS)
const GROUP = uppercase(
    if isnothing(arg_id)
        uppercase(get(ENV, "GROUP", "ALL"))
    else
        uppercase(only(match(pat, ARGS[arg_id]).captures))
    end,
)

@time begin
    if GROUP == "ALL" || GROUP == "VECTORSPACES"
        @time @safetestset "SumSpace" begin
            include("vectorspaces/sumspace.jl")
        end
    end

    if GROUP == "ALL" || GROUP == "ABSTRACTTENSOR"
        @time @safetestset "Indexing" begin
            include("abstracttensor/indexing.jl")
        end
        @time @safetestset "BlockTensor" begin
            include("abstracttensor/blocktensor.jl")
        end
        @time @safetestset "SparseBlockTensor" begin
            include("abstracttensor/sparseblocktensor.jl")
        end
        @time @safetestset "BraidingTensor" begin
            include("abstracttensor/braidingtensor.jl")
        end
    end

    if GROUP == "ALL" || GROUP == "LINALG"
        @time @safetestset "VectorInterface" begin
            include("linalg/vectorinterface.jl")
        end
        @time @safetestset "indexmanipulations" begin
            include("linalg/indexmanipulations.jl")
        end
        @time @safetestset "TensorOperations" begin
            include("linalg/tensoroperations.jl")
        end
        @time @safetestset "factorizations" begin
            include("linalg/factorizations.jl")
        end
    end

    if GROUP == "ALL" || GROUP == "UTILITY"
        @time @safetestset "aqua" begin
            include("aqua.jl")
        end
    end
end
