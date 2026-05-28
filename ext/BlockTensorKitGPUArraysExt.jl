module BlockTensorKitGPUArraysExt

using BlockTensorKit, BlockArrays, GPUArrays, Strided
using Strided: StridedViews
using GPUArrays: KernelAbstractions

function KernelAbstractions.get_backend(BA::BlockArrays.BlockArray{T, N, A}) where {T, N, A <: AbstractArray{<:StridedView{T, N, <:AnyGPUArray}}}
    return KernelAbstractions.get_backend(first(BA.blocks))
end

function Base.copyto!(dest::BM, src::TA) where {T <: Number, TA <: AnyGPUMatrix{T}, BM <: BlockMatrix{T, Matrix{TA}}}
    # TODO -- should we use Threads here to parallelize these
    # transfers in streams if possible?
    for block_index in Iterators.product(blockaxes(dest)...)
        indices = getindex.(axes(dest), block_index)
        dest_view = @view dest[block_index...]
        dest_view .= src[indices...]
    end
    return dest
end

end
