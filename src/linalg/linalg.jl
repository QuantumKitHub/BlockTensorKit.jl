# Linear Algebra
# --------------
Base.:(+)(t::AbstractBlockTensorMap, t2::AbstractBlockTensorMap) = add(t, t2)
function Base.:(-)(t::AbstractBlockTensorMap, t2::AbstractBlockTensorMap)
    return add(t, t2, -one(scalartype(t)))
end
Base.:(*)(t::AbstractBlockTensorMap, α::Number) = scale(t, α)
Base.:(*)(α::Number, t::AbstractBlockTensorMap) = scale(t, α)
Base.:(/)(t::AbstractBlockTensorMap, α::Number) = scale(t, inv(α))
Base.:(\)(α::Number, t::AbstractBlockTensorMap) = scale(t, inv(α))

function LinearAlgebra.axpy!(α::Number, t1::BlockTensorMap, t2::BlockTensorMap)
    space(t1) == space(t2) || throw(SpaceMismatch())
    for (i, v) in nonzero_pairs(t1)
        t2[i] = axpy!(α, v, t2[i])
    end
    return t2
end

function LinearAlgebra.axpby!(α::Number, t1::BlockTensorMap, β::Number, t2::BlockTensorMap)
    space(t1) == space(t2) || throw(SpaceMismatch())
    rmul!(t2, β)
    for (i, v) in nonzero_pairs(t1)
        t2[i] = axpy!(α, v, t2[i])
    end
    return t2
end

function LinearAlgebra.dot(t1::BlockTensorMap, t2::BlockTensorMap)
    size(t1) == size(t2) || throw(DimensionMismatch("dot arguments have different size"))

    s = zero(promote_type(scalartype(t1), scalartype(t2)))
    if nonzero_length(t1) >= nonzero_length(t2)
        @inbounds for (I, v) in nonzero_pairs(t1)
            s += dot(v, t2[I])
        end
    else
        @inbounds for (I, v) in nonzero_pairs(t2)
            s += dot(t1[I], v)
        end
    end
    return s
end

function LinearAlgebra.mul!(C::BlockTensorMap, α::Number, A::BlockTensorMap)
    space(C) == space(A) || throw(SpaceMismatch())
    SparseArrayKit._zero!(parent(C))
    for (i, v) in nonzero_pairs(A)
        C[i] = mul!(C[i], α, v)
    end
    return C
end

for TA in (:AbstractBlockTensorMap, :(TK.DiagonalTensorMap), :(TK.AdjointTensorMap), :TensorMap),
        TB in (:AbstractBlockTensorMap, :(TK.DiagonalTensorMap), :(TK.AdjointTensorMap), :TensorMap)
    (TA === :AbstractBlockTensorMap || TB === :AbstractBlockTensorMap) || continue
    @eval function TK.compose_dest(A::$TA, B::$TB)
        S = TK.check_spacetype(A, B)
        TC = TO.promote_contract(scalartype(A), scalartype(B), One)
        M = TK.promote_storagetype(TK.similarstoragetype(A, TC), TK.similarstoragetype(B, TC))
        TTC = if issparse(A) && issparse(B)
            sparseblocktensormaptype(S, numout(A), numin(B), M)
        else
            blocktensormaptype(S, numout(A), numin(B), M)
        end
        structure = codomain(A) ← domain(B)
        return TO.tensoralloc(TTC, structure, Val(false))
    end
end

# This is a generic implementation of `mul!` for BlockTensors that is used to make it easier
# to work with abstract element types, that might not support in-place operations.
# For now, the implementation might not be hyper-optimized, but the assumption is that we
# are in the limit where multiplying the individual elements is the bottleneck.
# With that in mind, we simply write the multiplication in terms of sparse tensors.
function LinearAlgebra.mul!(
        C::AbstractBlockTensorMap, A::AbstractBlockTensorMap, B::AbstractBlockTensorMap,
        α::Number, β::Number,
    )
    compose(space(A), space(B)) == space(C) ||
        throw(SpaceMismatch(lazy"$(space(C)) ≠ $(space(A)) * $(space(B))"))

    scale!(C, β)

    sortIA(IA) = CartesianIndex(TT.getindices(IA.I, domainind(A)))
    keysA = sort!(vec(collect(nonzero_keys(A))); by = sortIA)
    sortIB(IB) = CartesianIndex(TT.getindices(IB.I, codomainind(B)))
    keysB = sort!(vec(collect(nonzero_keys(B))); by = sortIB)

    iA = iB = 1
    @inbounds while iA <= length(keysA) && iB <= length(keysB)
        IA = keysA[iA]
        IB = keysB[iB]
        IAc = CartesianIndex(TT.getindices(IA.I, domainind(A)))
        IBc = CartesianIndex(TT.getindices(IB.I, codomainind(B)))
        if IAc == IBc
            Ic = IAc
            jA = iA
            while jA < length(keysA)
                if CartesianIndex(TT.getindices(keysA[jA + 1].I, domainind(A))) == Ic
                    jA += 1
                else
                    break
                end
            end
            jB = iB
            while jB < length(keysB)
                if CartesianIndex(TT.getindices(keysB[jB + 1].I, codomainind(B))) == Ic
                    jB += 1
                else
                    break
                end
            end
            rA = iA:jA
            rB = iB:jB
            if length(rA) < length(rB)
                for kB in rB
                    IB = keysB[kB]
                    IBo = CartesianIndex(TT.getindices(IB.I, domainind(B)))
                    vB = B[IB]
                    for kA in rA
                        IA = keysA[kA]
                        IAo = CartesianIndex(TT.getindices(IA.I, codomainind(A)))
                        IABo = CartesianIndex(IAo, IBo)
                        IC = CartesianIndex(TT.getindices(IABo.I, allind(C)))
                        vA = A[IA]
                        increasemulindex!(C, vA, vB, α, One(), IC)
                    end
                end
            else
                for kA in rA
                    IA = keysA[kA]
                    IAo = CartesianIndex(TT.getindices(IA.I, codomainind(A)))
                    vA = A[IA]
                    for kB in rB
                        IB = keysB[kB]
                        IBo = CartesianIndex(TT.getindices(IB.I, domainind(B)))
                        vB = B[IB]
                        IABo = CartesianIndex(IAo, IBo)
                        IC = CartesianIndex(TT.getindices(IABo.I, allind(C)))
                        increasemulindex!(C, vA, vB, α, One(), IC)
                    end
                end
            end
            iA = jA + 1
            iB = jB + 1
        elseif IAc < IBc
            iA += 1
        else
            iB += 1
        end
    end

    return C
end

@inline function increasemulindex!(
        C::AbstractBlockTensorMap, A::AbstractTensorMap, B::AbstractTensorMap,
        α::Number, β::Number, I,
    )
    return if haskey(C, I)
        C[I] = _mul!!(C[I], A, B, α, β)
    else
        C[I] = _mul!!(nothing, A, B, α, β)
    end
end

_mul!!(::Nothing, A, B, α::Number, β::Number) = scale!!(A * B, α)
_mul!!(C, A, B, α::Number, β::Number) = add!!(C, A * B, α, β)
const _TM_CAN_MUL = Union{
    TensorMap, AdjointTensorMap{<:Any, <:Any, <:Any, <:Any, <:TensorMap}, BraidingTensor,
}
function _mul!!(C::_TM_CAN_MUL, A::_TM_CAN_MUL, B::_TM_CAN_MUL, α::Number, β::Number)
    return mul!(C, A, B, α, β)
end
# TODO: optimize other implementations

# ensure that mixes with AbstractBlockTensorMap and AbstractTensorMap behave as expected:
for (TC, TA, TB) in Iterators.product(
        Iterators.repeated((:AbstractTensorMap, :AbstractBlockTensorMap), 3)...
    )
    (:AbstractBlockTensorMap ∉ (TC, TA, TB) || all(==(:AbstractBlockTensorMap), (TC, TA, TB))) && continue
    @eval function LinearAlgebra.mul!(C::$TC, A::$TA, B::$TB, α::Number, β::Number)
        A′ = A isa AbstractBlockTensorMap ? A : convert(BlockTensorMap, A)
        B′ = B isa AbstractBlockTensorMap ? B : convert(BlockTensorMap, B)
        if C isa AbstractBlockTensorMap
            return mul!(C, A′, B′, α, β)
        else
            C′ = convert(BlockTensorMap, C)
            C′ = mul!(C′, A′, B′, α, β)
            return only(C′)
        end
    end
end

function LinearAlgebra.norm(tA::BlockTensorMap, p::Real = 2)
    vals = nonzero_values(tA)
    isempty(vals) && return norm(zero(scalartype(tA)), p)
    return LinearAlgebra.norm(norm.(vals), p)
end

for f in (:real, :imag)
    @eval function Base.$f(t::AbstractBlockTensorMap)
        if isreal(sectortype(spacetype(t)))
            TT = Base.promote_op($f, eltype(t))
            t′ = similar(t, TT)
            @inbounds for (k, v) in nonzero_pairs(t)
                t′[k] = $f(v)
            end
            return t′
        else
            msg = "`$f` has not been implemented for `BlockTensorMap{$(S)}`."
            throw(ArgumentError(msg))
        end
    end
end

function TK.:(⊗)(t1::AbstractBlockTensorMap, t2::AbstractBlockTensorMap)
    (S = spacetype(t1)) === spacetype(t2) ||
        throw(SpaceMismatch("spacetype(t1) ≠ spacetype(t2)"))
    pA = ((codomainind(t1)..., TT.reverse(domainind(t1))...), ())
    pB = ((), (codomainind(t2)..., TT.reverse(domainind(t2))...))
    pAB = (
        (codomainind(t1)..., (codomainind(t2) .+ numind(t1))...),
        (
            TT.reverse(domainind(t1) .+ numout(t1))...,
            TT.reverse(domainind(t2) .+ (numind(t1) + numout(t2)))...,
        ),
    )
    return tensorproduct(t1, pA, false, t2, pB, false, pAB)
end

function LinearAlgebra.isposdef!(t::AbstractBlockTensorMap)
    domain(t) == codomain(t) ||
        throw(SpaceMismatch("`isposdef` requires domain and codomain to be the same"))
    InnerProductStyle(spacetype(t)) === EuclideanInnerProduct() || return false
    for (c, b) in TK.blocks(t)
        isposdef!(b) || return false
    end
    return true
end

function LinearAlgebra.lmul!(D::DiagonalTensorMap, t::AbstractBlockTensorMap)
    domain(D) == codomain(t) || throw(SpaceMismatch())
    TensorKit.foreachblock(t, D) do c, (tblock, bs...)
        tblock′ = lmul!(bs..., copy_dense!(similar_dense(tblock), tblock))
        tblock === tblock′ || copyto!(tblock, tblock′)
        return tblock
    end
    return t
end

function LinearAlgebra.rmul!(t::AbstractBlockTensorMap, D::DiagonalTensorMap)
    codomain(D) == domain(t) || throw(SpaceMismatch())
    TensorKit.foreachblock(t, D) do c, (tblock, bs...)
        tblock′ = rmul!(copy_dense!(similar_dense(tblock), tblock), bs...)
        tblock === tblock′ || copyto!(tblock, tblock′)
        return tblock
    end
    return t
end
