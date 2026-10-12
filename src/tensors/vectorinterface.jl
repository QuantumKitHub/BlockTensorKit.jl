# zerovector
# ----------
VI.zerovector!(t::AbstractBlockTensorMap) = (zerovector!(parent(t)); t)
VI.zerovector!(t::SparseBlockTensorMap) = (empty!(t.data); t)

# scale
# -----
function VI.scale!(t::AbstractBlockTensorMap, α::Number)
    foreach(Base.Fix2(scale!, α), nonzero_values(t))
    return t
end

VI.scale!(ty::AbstractBlockTensorMap, tx::AbstractBlockTensorMap, α::Number) =
    add!(ty, tx, α, Zero())

# add
# ---
function VI.add(ty::AbstractBlockTensorMap, tx::AbstractBlockTensorMap, α::Number, β::Number)
    S = TK.check_spacetype(ty, tx)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))

    # result type defaults to TensorMap if the types don't match to avoid assymmetric
    # implementation via zerovector(ty, T) vs zerovector(tx, T)
    # This would give issues for example with DiagonalTensorMap + TensorMap
    T = VectorInterface.promote_add(ty, tx, α, β)
    tdst = if typeof(ty) === typeof(tx)
        similar(ty, T)
    else
        M = TK.promote_storagetype(TK.similarstoragetype(ty, T), TK.similarstoragetype(tx, T))
        if issparse(ty) && issparse(tx)
            sparseblocktensormaptype(S, numout(ty), numin(ty), M)(undef, space(ty))
        else
            blocktensormaptype(S, numout(ty), numin(ty), M)(undef, space(ty))
        end
    end

    return add!(scale!(tdst, ty, β), tx, α)
end

function VI.add!(ty::AbstractBlockTensorMap, tx::AbstractBlockTensorMap, α::Number, β::Number)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    _scale_untouched!(ty, tx, β)
    for (I, v) in nonzero_pairs(tx)
        _addblock!((y, γ) -> add!!(y, v, α, γ), ty, I, β)
    end
    return ty
end

# blockwise updates
# -----------------
@inline function _addblock!(
        f, t::AbstractBlockTensorMap, I::CartesianIndex, β::Number = One(), allocator = TO.DefaultAllocator()
    )
    t[I] = if haskey(t, I)
        f(t[I], β)
    else
        f(TO.tensoralloc(eltype(t), eachspace(t)[I], Val(false), allocator), Zero())
    end
    return t
end

function _scale_untouched!(
        t::AbstractBlockTensorMap, tsrc::AbstractTensorMap, β::Number, p::IndexTuple = ntuple(identity, numind(t))
    )
    (issparse(tsrc) && !isone(β)) || return t
    q = TT.invperm(p)
    istouched(I) = haskey(tsrc, CartesianIndex(TT.getindices(I.I, q)))
    if issparse(t) && iszero(β)
        filter!(istouched ∘ first, parent(t))
    else
        for (I, v) in nonzero_pairs(t)
            istouched(I) || scale!(v, β)
        end
    end
    return t
end

# inner
# -----
function VI.inner(x::AbstractBlockTensorMap, y::AbstractBlockTensorMap)
    space(x) == space(y) || throw(SpaceMismatch("$(space(x)) ≠ $(space(y))"))
    T = VI.promote_inner(x, y)
    # only entries that are nonzero in both contribute
    ks = if issparse(x) && issparse(y)
        intersect(nonzero_keys(x), nonzero_keys(y))
    else
        nonzero_keys(issparse(y) ? y : x)
    end
    return sum(ks; init = zero(T)) do I
        inner(x[I], y[I])
    end
end
