# Scalar Spherical Harmonics as basis functions on the sphere for a spectral
# method for solving PDEs

#=
NOTE

    OPs: Y^m_l(x,y,z) ≡ Y^m_l(θ,z) := P^{(m,m)}_{l-m}(z) * ρ(z)^m * exp(miθ)

for l ∈ ℕ₀, m = -l,...,l

where x = cosθ sinϕ, y = sinθ sinϕ, z = cosϕ; ρ(z) := sqrt(1-z^2) = sinϕ

=#

export SphericalHarmonicsFamily, SphericalHarmonicsSpace, SphericalHarmonicsTangentSpace

abstract type SphericalFamily{B,T} end
struct SphereSurface{B,T} <: Domain{SVector{2,B}} end
SphereSurface() = SphereSurface{BigFloat, Float64}()
function checkpoints(::SphereSurface)
    # Return 2 3D points that will lie on the domain TODO
    y, z = -0.3, 0.8; x = sqrt(1 - z^2 - y^2); p1 = SVector(x, y, z)
    y, z = 0.4, 0.8; x = sqrt(1 - z^2 - y^2); p2 = SVector(x, y, z)
    [p1, p2]
end
in(x::SVector{3}, D::SphereSurface) =
    D.α ≤ x[3] ≤ D.β && sqrt(x[1]^2 + x[2]^2) == D.ρ(x[3])


# Structs of the spaces
struct SphericalHarmonicsSpace{DF, B, T} <: Space{SphereSurface{B,T}, T}
    family::DF # Pointer back to the family
    opptseval::Vector{Vector{B}} # Store the ops evaluated at the transform pts
    A::Vector{Vector{SparseMatrixCSC{B}}}
    B::Vector{Vector{SparseMatrixCSC{B}}}
    C::Vector{Vector{SparseMatrixCSC{B}}}
    DT::Vector{Vector{SparseMatrixCSC{B}}}
end

struct SphericalHarmonicsTangentSpace{DF, B, T} <: Space{SphereSurface{B,T}, T}
    family::DF # Pointer back to the family
    A::Vector{Vector{BandedBlockBandedMatrix{B}}}
    B::Vector{Vector{BandedBlockBandedMatrix{B}}}
    C::Vector{Vector{BandedBlockBandedMatrix{B}}}
    DT::Vector{Vector{SparseMatrixCSC{B}}}
end

function SphericalHarmonicsSpace(fam::SphericalFamily{B,T}) where {B,T}
    SphericalHarmonicsSpace{typeof(fam), B, T}(
        fam,
        Vector{Vector{B}}(),
        Vector{Vector{SparseMatrixCSC{B}}}(),
        Vector{Vector{SparseMatrixCSC{B}}}(),
        Vector{Vector{SparseMatrixCSC{B}}}(),
        Vector{Vector{SparseMatrixCSC{B}}}())
end

spacescompatible(A::SphericalHarmonicsSpace, B::SphericalHarmonicsSpace) = true

domain(::SphericalHarmonicsSpace{<:Any,B,T}) where {B,T} = SphereSurface{B,T}()

struct SphericalHarmonicsFamily{B,T,F} <: SphericalFamily{B,T}
    space::Vector{SphericalHarmonicsSpace}
    tangentspace::Vector{SphericalHarmonicsTangentSpace}
    α::T
    β::T
    ρ::F # Fun of sqrt(1-X^2) in (α,β)
    P::OrthogonalPolynomialFamily # Family for weight ρ² in [-1,1] (Jacobi)
end

function (D::SphericalHarmonicsFamily)()
    if length(D.space) == 1
        D.space[1]
    elseif length(D.space) == 0
        resize!(D.space, 1)
        D.space[1] = SphericalHarmonicsSpace(D)
    else
        error("space should be a vector of length 1 or 0")
    end
end

function SphericalHarmonicsFamily(::Type{B}, ::Type{T}) where {B,T}
    β = T(1)
    α = -β
    X = Fun(identity, α..β)
    ρ = sqrt(1 - X^2)
    P = OrthogonalPolynomialFamily(T, 1-X, 1+X)
    SphericalHarmonicsFamily{B,T,typeof(ρ)}(
        Vector{SphericalHarmonicsSpace}(),
        Vector{SphericalHarmonicsTangentSpace}(),
        α, β, ρ, P)
end
# Useful quick constructors
SphericalHarmonicsFamily() = SphericalHarmonicsFamily(ComplexF64, Float64)
# SphericalHarmonicsFamily() = SphericalHarmonicsFamily(Float64, Float64)


#===#
# Normalising constant calculation method
function getnormalisingconstant(::Type{T}, S::SphericalHarmonicsSpace, l::Int,
                                m::Int) where T
    """ Returns clm for the Ylm SH OP for valid l, m pair

    Returns as type T.

    Commented out is the written way of calculating. The factorials can get VERY
    large and so cannot be computed, so we can implement a better algorithm to
    deal with them.
    """
    # f1 = factorial(l - m)
    # f2 = factorial(l + m)
    # f3 = factorial(l)
    # ret = sqrt(T.(2l + 1) * f1 / (4π * f2))
    # ret *= f2 / (2^(abs(m)) * f3)
    # if m ≥ 0 && isodd(m)
    #     ret *= -1
    # end
    # T(ret)

    @assert l ≥ abs(m) "Invalid l, m pair given - l should be ≥ |m|"
    absm = abs(m)
    ret = T(1)
    ret *= 2l + 1
    if absm > 0
        for k = 1:absm
            ret *= l + k
            ret /= l - absm + k
        end
    end
    ret = sqrt(ret)
    ret /= 2^absm * 2 * sqrt(π)
    if m ≥ 0 && isodd(m)
        ret *= -1
    end
    T(ret)
end

function getdegzeropteval(::Type{T}, S::SphericalHarmonicsSpace) where T
    """ Returns the constant that is Y00 (the degree zero SH). """
    T(1) / sqrt(4π)
end

#===#
# Recurrence coefficients/Jacobi matrix entries
function getrecα̃(::Type{T}, S::SphericalHarmonicsSpace, li::Int, mi::Int, j::Int) where T
    """ Returns the α̃_{l,m,j} value for j = 1,2,3,4 """

    # We want to use T arithmetic, so convert. We would rather the inputs be
    # Ints though
    l = T(li); m = T(mi)
    ret = 0
    if j == 1
        ret += 2l / (2l+1)
    elseif j == 2
        if li - mi ≥ 2
            ret -= l / (2(2l+1))
        end
    elseif j == 3
        ret -= 2(l-m+2) * (l-m+1) / ((2l+1) * (l+1))
    elseif j == 4
        ret += (l+m+2) * (l+m+1) / (2(2l+1) * (l+1))
    else
        error("Invalid α̃ coeff being requested")
    end
    ret
end
function getrecγ̃(::Type{T}, S::SphericalHarmonicsSpace, li::Int, mi::Int, j::Int) where T
    """ Returns the γ̃_{l,m,j} value for j = 1,2 """

    # We want to use T arithmetic, so convert. We would rather the inputs be
    # Ints though
    l = T(li); m = T(mi)
    ret = 0
    if j == 1
        if li - mi ≥ 1
            ret += l / (2l+1)
        end
    elseif j == 2
        ret += (l-m+1) * (l+m+1) / ((2l+1) * (l+1))
    else
        error("Invalid γ̃ coeff being requested")
    end
    ret
end
function recα(::Type{T}, S::SphericalHarmonicsSpace, l::Int, m::Int, j::Int) where T
    """ Returns the mult by x coeff, α_{l,m,j}, value for j = 1,2,3,4 """

    ret = 0
    if j == 1
        if m > 0
            ret += getrecα̃(T, S, l, m, 1)
            ret *= getnormalisingconstant(T, S, l, m) / (2getnormalisingconstant(T, S, l-1, m-1))
        elseif l - abs(m) ≥ 2
            ret += getrecα̃(T, S, l, abs(m), 2)
            ret *= getnormalisingconstant(T, S, l, m) / (2getnormalisingconstant(T, S, l-1, m-1))
        end
    elseif j == 2
        if m < 0
            ret += getrecα̃(T, S, l, abs(m), 1)
            ret *= getnormalisingconstant(T, S, l, m) / (2getnormalisingconstant(T, S, l-1, m+1))
        elseif l - m ≥ 2
            ret += getrecα̃(T, S, l, m, 2)
            ret *= getnormalisingconstant(T, S, l, m) / (2getnormalisingconstant(T, S, l-1, m+1))
        end
    elseif j == 3
        if m > 0
            ret += getrecα̃(T, S, l, m, 3)
            ret *= getnormalisingconstant(T, S, l, m) / (2getnormalisingconstant(T, S, l+1, m-1))
        else
            ret += getrecα̃(T, S, l, abs(m), 4)
            ret *= getnormalisingconstant(T, S, l, m) / (2getnormalisingconstant(T, S, l+1, m-1))
        end
    elseif j == 4
        if m < 0
            ret += getrecα̃(T, S, l, abs(m), 3)
            ret *= getnormalisingconstant(T, S, l, m) / (2getnormalisingconstant(T, S, l+1, m+1))
        else
            ret += getrecα̃(T, S, l, m, 4)
            ret *= getnormalisingconstant(T, S, l, m) / (2getnormalisingconstant(T, S, l+1, m+1))
        end
    else
        error("Invalid α or β coeff being requested")
    end
    ret
end
function recβ(::Type{T}, S::SphericalHarmonicsSpace, l::Int, m::Int, j::Int) where T
    """ Returns the mult by y coeff, β_{l,m,j}, value for j = 1,2,3,4 """

    (-1)^(j+1) * im * recα(T, S, l, m, j)
end
function recγ(::Type{T}, S::SphericalHarmonicsSpace, l::Int, m::Int, j::Int) where T
    """ Returns the mult by z coeff, γ_{l,m,j}, value for j = 1,2 """

    @assert (j == 1 || j == 2) "Invalid γ coeff being requested"
    ret = getrecγ̃(T, S, l, m, j) * getnormalisingconstant(T, S, l, m)
    if j == 1
        ret /= getnormalisingconstant(T, S, l-1, m)
    else
        ret /= getnormalisingconstant(T, S, l+1, m)
    end
    ret
end


#===#
# Indexing retrieval methods
function getopindex(S::SphericalHarmonicsSpace, l::Int, m::Int)
    """ Method to return the index (of a vector) corresponding to the Ylm SH OP
    """

    @assert abs(m) ≤ l "Invalid inputs to getopindex"
    l^2 + l + m + 1
end
function getlm(S::SphericalHarmonicsSpace, ind::Int)
    """ Method to return the corresponding SH OP (l,m orders) given the index
    (of a vector)
    """

    # TODO check this
    l = 0
    while true
        if (l+1)^2 ≥ ind
            break
        end
        l += 1
    end
    remainder = ind - l^2
    m = - l + remainder - 1
    l, m
end


#===#
# points()

function pointswithweights(S::SphericalHarmonicsSpace{<:Any, B, T}, n::Int) where {B,T}
    """

    # Return the weights and nodes to use for the even part of a function (poly),
    # i.e. for the sphere Ω:
    #   int_Ω f(x,y,z) dA = int_{-1}^1 int_0^{2pi} f(ρcosθ,ρsinθ,z) ρ dz dθ
            ≈ 0.5 * Σ_j wⱼ*[f(xⱼ,yⱼ,zⱼ) + f(-xⱼ,-yⱼ,zⱼ)]
    # NOTE: the "odd" part of the quad rule will equal 0 for polynomials,
    #       so can be ignored.

    # NOTE we output M≈n points (x,y,z), plus the M≈n points corresponding to
    (-x,-y,z)
    """

    @assert n > 1 "At least 1 point needs to be asked for in pointswithweights()."

    N = Int(ceil(-1.5 + 0.5 * sqrt(9 - 4 * (2 - 2n)))) # Degree of f
    M1 = Int(ceil((N+1) / 2)) # Quad rule for z interval exact for degree polynomial up to 2M1 - 1
    M2 = N + 1 # Quad rule for circle exact for polynomial up to degree M2 - 1
    M = M1 * M2 # Quad rule on Ω is exact for polynomials of degree N s.t. we have M points
    @show "begin pointswithweights()", n, M, N

    # Get the 1D quadrature pts and weights
    t, wt = OrthogonalPolynomialFamilies.pointswithweights(T, S.family.P(T(0), T(0)), M1) # Quad rule for [-1,1] interval
    s = [(cospi(T(2 * (it - 1)) / M2), sinpi(T(2 * (it - 1)) / M2)) for it=1:M2]
    ws = ones(T, M2) * 2π / M2  # Quad for circumference of unit circle

    # output both pts and the weights that make up the 3D quad rule
    pts = Vector{SArray{Tuple{3},T,1,3}}(undef, 2M) # stores both (x,y,z) and (-x,-y,z)
    w = zeros(T, M)
    for j1 = 1:M1
        z = t[j1]
        rhoz = S.family.ρ(z)
        for j2 = 1:M2
            x, y = rhoz * s[j2][1], rhoz * s[j2][2]
            pts[j2 + (j1 - 1)M2] = x, y, z
            pts[M + j2 + (j1 - 1)M2] = -x, -y, z
            w[j2 + (j1 - 1)M2] = wt[j1] * ws[j2]
        end
    end

    @show "end pointswithweights()"
    pts, w
end
points(S::SphericalHarmonicsSpace, n::Int) = pointswithweights(S, n)[1]


#===#
# Point evaluation of the OPs TODO

# Methods to gather and evaluate the ops of space S at the transform pts given
function getopptseval(S::SphericalHarmonicsSpace, N::Int, pts)
    """ Stores and returns the vals of the OPs evaluated at the points pts up to
    and including degree N.

    """

    resetopptseval(S)
    jj = [getopindex(S, l, -l) for l = 0:N]
    for j in jj
        opevalatpts(S, j, pts)
    end
    S.opptseval
end
function resetopptseval(S::SphericalHarmonicsSpace)
    """ Resets (makes length 0) the opptseval member of the
    SphericalHarmonicsSpace object.

    """

    resize!(S.opptseval, 0)
    S
end
function opevalatpts(S::SphericalHarmonicsSpace{<:Any, B, T}, j::Int, pts) where {B,T}
    """ NOTE This function should be used only from getopptseval().
    #      The idea here is that we have all OPs up to degree N already
    #      evaluated at pts, and we simply iterate once to calculate the pts
    #      evals for the deg N+1 OPs.
    # The input j refers to the index of a deg N+1 OP, that can be used to
    # return the vals for the OP.

    n is used in place of l here.

    """

    # If we already have the vals, return
    len = length(S.opptseval)
    if len ≥ j
        return S.opptseval[j]
    end

    # We iterate up from the last obtained pts eval, so check its been done
    N = len == 0 ? -1 : getlm(S, len)[1]
    n = getlm(S, j)[1]
    if  N != n - 1 || (len == 0 && j > 1)
        error("Invalid index")
    end

    # Calculate the vals using the building OPs formula (using reccurence mats)
    if n == 0
        # n == 0 is special case
        resize!(S.opptseval, 1)
        S.opptseval[1] = Vector{B}(undef, length(pts))
        S.opptseval[1][:] .= getdegzeropteval(B, S)
    else
        jj = getopindex(S, n, -n)
        resizedata!(S, n)
        resize!(S.opptseval, getopindex(S, n, n))
        for k = 0:2n
            S.opptseval[jj+k] = Vector{B}(undef, length(pts))
        end
        if n == 1
            for r = 1:length(pts)
                x, y, z = pts[r]
                ptsevalforopblock = getptsevalforop(S, n-1; blck=true)
                P1 = [ptsevalforopblock[it][r] for it = 1:2(n-1)+1]
                P = - clenshawDTBmG(S, n-1, P1, x, y, z; clenshawalg=false)
                for k = 0:2n
                    S.opptseval[jj+k][r] = P[k+1]
                end
            end
            # nm1 = getopindex(S, n-1, 1-n)
            # for r = 1:length(pts)
            #     x, y, z = pts[r]
            #     P1 = [opevalatpts(S, nm1+it, pts)[r] for it = 0:2(n-1)]
            #     P = - clenshawDTBmG(S, n-1, P1, x, y, z; clenshawalg=false)
            #     for k = 0:2n
            #         S.opptseval[jj+k][r] = P[k+1]
            #     end
            # end
        else
            for r = 1:length(pts)
                x, y, z = pts[r]
                ptsevalforopblock1 = getptsevalforop(S, n-1; blck=true)
                ptsevalforopblock2 = getptsevalforop(S, n-2; blck=true)
                P1 = [ptsevalforopblock1[it][r] for it = 1:2(n-1)+1]
                P2 = [ptsevalforopblock2[it][r] for it = 1:2(n-2)+1]
                P = - (clenshawDTBmG(S, n-1, P1, x, y, z; clenshawalg=false)
                        + clenshawDTC(S, n-1, P2; clenshawalg=false))
                for k = 0:2n
                    S.opptseval[jj+k][r] = P[k+1]
                end
            end
            # nm1 = getopindex(S, n-1, 0, 0)
            # nm2 = getopindex(S, n-2, 0, 0)
            # for r = 1:length(pts)
            #     x, y, z = pts[r]
            #     P1 = [opevalatpts(S, nm1+it, pts)[r] for it = 0:2(n-1)]
            #     P2 = [opevalatpts(S, nm2+it, pts)[r] for it = 0:2(n-2)]
            #     P = - (clenshawDTBmG(S, n-1, P1, x, y, z; clenshawalg=false)
            #             + clenshawDTC(S, n-1, P2; clenshawalg=false))
            #     for k = 0:2n
            #         S.opptseval[jj+k][r] = P[k+1]
            #     end
            # end
        end
    end
    S.opptseval[j]
end
# These funcs returns the S.opptseval for the OP n,k
# We assume that we have called getopptseval(S, N, pts) for large enough N
function getptsevalforop(S::SphericalHarmonicsSpace, ind::Int; blck::Bool=false)
    """ Returns the vals for the OP specified by the index ind. If blck=true,
    we return the vals for subblock of the OPs (of degree ind).

    # Assumes that we have called getopptseval(S, N, pts) for large enough N.
    """

    if blck
        N = getlm(S, length(S.opptseval))[1]
        if length(S.opptseval) != (N+1)^2
            error("Invalid length - getopptseval(S,N,pts) may not have been correctly called")
        else
            opptsevalpseudo = PseudoBlockArray(S.opptseval, [2l+1 for l=0:N])
            # the expession in the if clause is a way of finding the number of blocks
            if length(blockaxes(opptsevalpseudo)[1]) ≤ ind
                error("Invalid OP block requested in getptsevalforop - getopptseval(S,N,pts) may not have been correctly called")
            else
                opptsevalpseudo[Block(ind+1)]
            end
        end
    else
        if length(S.opptseval) < ind
            error("Invalid OP requested in getptsevalforop - getopptseval(S,N,pts) may not have been correctly called")
        else
            S.opptseval[ind]
        end
    end
end
getptsevalforop(S::SphericalHarmonicsSpace, l::Int, m::Int) =
    getptsevalforop(S, getopindex(S, l, m); blck=false)


#===#
# transform and itransform

struct SphericalHarmonicsTransformPlan{T}
    Vp::Array{T}
    Vm::Array{T}
end

function SphericalHarmonicsTransformPlan(S::SphericalHarmonicsSpace{<:Any, B, T},
                                            vals::AbstractVector{T}) where {B,T}
    """ Returns the 'plan' for the transform from spectral space to coefficient
    space.

    The plan consists of the Vandemonde matrices that are applied to the vector
    'vals' (the function evaluated at the quad points) to gain a vector of
    coefficients for said function in the OP basis.

    The entries to the matrices are the quad rule weights multiplied by the
    conjugate of the Ylm OP at the point x_j. Note that conj(Ylm) = (-1)^m Yl-m.

    Note for real valued functions, the coefficients flm satisfy
        f_{l,-m} = conj(f_{l,m})
    and so this could be simplified / made more efficient.

    # NOTE N here is the degree of the function f that we are finding the
    #      coefficients for.
    #      We should have M vals such that the quadrature rule is exact to
    #      calculate integrals of f * Ylm for l=0:N, which will have a max
    #      degree of 2N (i.e. N is constrained by the number of vals we have -
    #      with M pts, we have a quad rule exact for a poly of 2N).
    #      nops is the number of OPs we require (that is, all OPs up to and
    #      including deg N, i.e. length of ℙ_N) which is (N+1)^2.

    N - degree/order of function f approximated to
    M/npts - number of vals/quad points
    """

    @show "Begin SphericalHarmonicsTransformPlan"

    npts = Int(length(vals) / 2) # = M

    # Divide by 2 as the quad rule is for ∫_Ω f(x,y,z)*Ylm(x,y,z) dA for
    # l=0,...,N where deg(f)=N
    N = Int(floor(ceil(-1.5 + 0.5 * sqrt(9 - 4 * (2 - 2npts))) / 2))
    nops = (N+1)^2
    @show N, npts, nops

    # calculate the Vandermonde matrix
    pts, w = pointswithweights(S, npts)
    Vp = zeros(B, nops, npts); Vm = zeros(B, nops, npts)
    ptssarray = Vector{SArray{Tuple{3},T,1,3}}(undef, 2)
    for j = 1:npts
        if j % 100 == 0
            @show j, npts
        end
        ptssarray[1] = pts[j]; ptssarray[2] = pts[j+npts]
        getopptseval(S, N, ptssarray)
        indv = 1
        for l = 0:N
            indp = 0
            v = getptsevalforop(S, l; blck=true)
            for m = -l:l
                fctr = isodd(m) ? -1 : 1
                # We account for the complex conjugate in the inner product by
                # iterating backwards over the ptsevals
                Vp[indv, j] = v[end-indp][1] * w[j] * fctr
                Vm[indv, j] = v[end-indp][2] * w[j] * fctr
                indv += 1; indp += 1
            end
        end
        resetopptseval(S)
    end

    SCTP = SphericalHarmonicsTransformPlan{B}(Vp, Vm)
    @show "End SphericalHarmonicsTransformPlan"
    SCTP
end
plan_transform(S::SphericalHarmonicsSpace, vals) = SphericalHarmonicsTransformPlan(S, vals)
transform(S::SphericalHarmonicsSpace, vals) = plan_transform(S, vals) * vals

function *(SCTP::SphericalHarmonicsTransformPlan{B}, vals::AbstractVector{T}) where {B,T}
    """ Mult function for the plan and the vals of a function.

    Inputs: Plan, f(pts) for desired f
    Output: Coeffs of the func f for its expansion in the OP basis
    """

    @show "Begin SCTP mult"
    n = Int(length(vals) / 2)
    ret = 0.5 * (SCTP.Vp * vals[1:n] + SCTP.Vm * vals[n+1:end])
    @show "End SCTP mult"
    ret
end

function itransform(S::SphericalHarmonicsSpace, cfs::AbstractVector{T}) where T
    """ Inverse transform to gain function vals at the quad points from the
    coefficients of said function.

    Inputs: OP space, coeffs of a function f for its expansion in the SH OPs
    Output: vals = {f(x_j)} where x_j are are the points(S,n)
    """

    @show "begin itransform"
    ncfs = length(cfs)
    N = Int(sqrt(ncfs)) - 1 # We have (N+1)^2 OPs (number of OPs deg ≤ N)
    npts = (2N+1) * (N+1) # = (2N+1)(2N+2)/2
    @show npts, N
    pts = points(S, npts)
    ret = zeros(T, 2npts)

    ptssarray = Vector{SArray{Tuple{3},T,1,3}}(undef, 2)
    for j = 1:npts
        if j % 100 == 0
            @show j, npts
        end
        ptssarray[1] = pts[j]; ptssarray[2] = pts[j+npts]
        getopptseval(S, N, ptssarray)
        indc = 1
        for l = 0:N, m = -l:l
            ret[j] += S.opptseval[indc][1] * cfs[indc]
            ret[j+npts] += S.opptseval[indc][2] * cfs[indc]
            indc += 1
        end

        # for l = 0:N
        #     indp = 1
        #     v = getptsevalforop(S, l; blck=true)
        #     for m = -l:l
        #         ret[j] += v[indp][1] * cfs[indc]
        #         ret[j+npts] += v[indp][2] * cfs[indc]
        #         indc += 1; indp += 1
        #     end
        # end
        resetopptseval(S)
    end

    @show "end itransform"
    ret
end




#===#
# Function evaluation (clenshaw)

function getclenshawsubblockx(S::SphericalHarmonicsSpace{<:Any, T, <:Any},
                                l::Int; subblock::String="A") where T
    """ Returns the Jacobi matrix subblock A_{x,l}, B_{x,l}, C_{x,l} """

    @assert subblock in ("A", "B", "C") "Invalid subblock given"
    @assert l ≥ 0 "Invalid l - should be non-negative integer"

    if subblock == "A"
        band = 1
        mat = spzeros(T, 2l+1, 2(l+band)+1)
        for i = 1:2l+1
            mat[i,i] = recα(T, S, l, -l+i-1, 3)
            mat[i,i+2] = recα(T, S, l, -l+i-1, 4)
        end
    elseif subblock == "B"
        band = 0
        mat = spzeros(T, 2l+1, 2(l+band)+1)
    else
        band = -1
        l == 0 && error("l needs to be > 0 when Clenshaw mat C requested")
        mat = spzeros(T, 2l+1, 2(l+band)+1)
        for i = 1:2l-1
            mat[i,i] = recα(T, S, l, -l+i-1, 2)
            mat[i+2,i] = recα(T, S, l, -l+i+1, 1)
        end
    end
    mat
end
function getclenshawsubblocky(S::SphericalHarmonicsSpace{<:Any, T, <:Any},
                                l::Int; subblock::String="A") where T
    """ Returns the Jacobi matrix subblock A_{y,l}, B_{y,l}, C_{y,l} """

    @assert subblock in ("A", "B", "C") "Invalid subblock given"
    @assert l ≥ 0 "Invalid l - should be non-negative integer"

    if subblock == "A"
        band = 1
        mat = spzeros(T, 2l+1, 2(l+band)+1)
        for i = 1:2l+1
            mat[i,i] = recβ(T, S, l, -l+i-1, 3)
            mat[i,i+2] = recβ(T, S, l, -l+i-1, 4)
        end
    elseif subblock == "B"
        band = 0
        mat = spzeros(T, 2l+1, 2(l+band)+1)
    else
        band = -1
        l == 0 && error("l needs to be > 0 when Clenshaw mat C requested")
        mat = spzeros(T, 2l+1, 2(l+band)+1)
        for i = 1:2l-1
            mat[i,i] = recβ(T, S, l, -l+i-1, 2)
            mat[i+2,i] = recβ(T, S, l, -l+i+1, 1)
        end
    end
    mat
end
function getclenshawsubblockz(S::SphericalHarmonicsSpace{<:Any, T, <:Any},
                                l::Int; subblock::String="A") where T
    """ Returns the Jacobi matrix subblock A_{z,l}, B_{z,l}, C_{z,l} """

    @assert subblock in ("A", "B", "C") "Invalid subblock given"
    @assert l ≥ 0 "Invalid l - should be non-negative integer"

    if subblock == "A"
        band = 1
        mat = spzeros(T, 2l+1, 2(l+band)+1)
        for i = 1:2l+1
            mat[i,i+1] = recγ(T, S, l, -l+i-1, 2)
        end
    elseif subblock == "B"
        band = 0
        mat = spzeros(T, 2l+1, 2(l+band)+1)
    else
        band = -1
        l == 0 && error("l needs to be > 0 when Clenshaw mat C requested")
        mat = spzeros(T, 2l+1, 2(l+band)+1)
        for i = 1:2l-1
            mat[i+1,i] = recγ(T, S, l, -l+i, 1)
        end
    end
    mat
end

# NOTE Each of these """ Computes and stores the Jacobi matrix blocks up to deg N """
function getBs!(S::SphericalHarmonicsSpace{<:Any, T, <:Any}, N, N₀) where T
    m = N₀
    resize!(S.B, N + 1)
    subblock = "B"
    for n = N:-1:m
        S.B[n+1] = Vector{SparseMatrixCSC{T}}(undef, 3)
        resize!(S.B[n+1], 3)
        S.B[n+1][1] = getclenshawsubblockx(S, n; subblock=subblock)
        S.B[n+1][2] = getclenshawsubblocky(S, n; subblock=subblock)
        S.B[n+1][3] = getclenshawsubblockz(S, n; subblock=subblock)
    end
    S
end
function getCs!(S::SphericalHarmonicsSpace{<:Any, T, <:Any}, N, N₀) where T
    m = N₀
    resize!(S.C, N + 1)
    subblock = "C"
    if N₀ == 0
        m += 1 # C_0 does not exist
    end
    for n = N:-1:m
        S.C[n+1] = Vector{SparseMatrixCSC{T}}(undef, 3)
        resize!(S.C[n+1], 3)
        S.C[n+1][1] = getclenshawsubblockx(S, n; subblock=subblock)
        S.C[n+1][2] = getclenshawsubblocky(S, n; subblock=subblock)
        S.C[n+1][3] = getclenshawsubblockz(S, n; subblock=subblock)
    end
    S
end
function getAs!(S::SphericalHarmonicsSpace{<:Any, T, <:Any}, N, N₀) where T
    m = N₀
    resize!(S.A, N + 1)
    subblock = "A"
    for n = N:-1:m
        S.A[n+1] = Vector{SparseMatrixCSC{T}}(undef, 3)
        resize!(S.A[n+1], 3)
        S.A[n+1][1] = getclenshawsubblockx(S, n; subblock=subblock)
        S.A[n+1][2] = getclenshawsubblocky(S, n; subblock=subblock)
        S.A[n+1][3] = getclenshawsubblockz(S, n; subblock=subblock)
    end
    S
end

function getDTs!(S::SphericalHarmonicsSpace{<:Any, T, <:Any}, N, N₀) where T
    """ Computes and stores Blocks that make up the matrix Dᵀ_l

    # Need to store these as BandedBlockBandedMatrices for each subblock
    # corresponding to x,y,z.
    # i.e. We store [DT_{x,n}, DT_{y,n}, DT_{z,n}] where
    #    I = DTn*An = DT_{x,n}*A_{x,n} + DT_{y,n}*A_{y,n} + DT_{z,n}*A_{z,n}
    """

    previousN = N₀
    resize!(S.DT, N + 1)
    if previousN == 0
        l = 0
        S.DT[l+1] = Vector{SparseMatrixCSC{T}}(undef, 3)
        resize!(S.DT[l+1], 3)
        denom = (recα(T, S, l, l, 3) * recβ(T, S, l, l, 4)
                    - recα(T, S, l, l, 4) * recβ(T, S, l, l, 3))

        S.DT[l+1][1] = spzeros(2l+3, 2l+1)
        S.DT[l+1][1][1,1] = recβ(T, S, l, l, 4) / denom
        S.DT[l+1][1][3,1] = - recβ(T, S, l, l, 3) / denom

        S.DT[l+1][2] = spzeros(2l+3, 2l+1)
        S.DT[l+1][2][1,1] = - recα(T, S, l, l, 4) / denom
        S.DT[l+1][2][3,1] = recα(T, S, l, l, 3) / denom

        S.DT[l+1][3] = spzeros(2l+3, 2l+1)
        S.DT[l+1][3][2,1] = 1 / recγ(T, S, l, l, 2)

        previousN += 1
    end
    for l = N:-1:previousN
        S.DT[l+1] = Vector{SparseMatrixCSC{T}}(undef, 3)
        resize!(S.DT[l+1], 3)

        # Define
        S.DT[l+1][1] = spzeros(2l+3, 2l+1)
        α3, α4 = recα(T, S, l, -l, 3), recα(T, S, l, l, 4)
        S.DT[l+1][1][1, 1] = 1 / α3
        S.DT[l+1][1][2l+3, 2l+1] = 1 / α4

        S.DT[l+1][2] = spzeros(2l+3, 2l+1)

        S.DT[l+1][3] = spzeros(2l+3, 2l+1)
        Dz = S.DT[l+1][3]
        Dz[1, 2] = - recα(T, S, l, -l, 4) / (α3 * recγ(T, S, l, -l+1, 2))
        m = -l
        for i = 1:2l+1
            Dz[i+1, i] = 1 / recγ(T, S, l, m, 2)
            m += 1
        end
        Dz[2l+3, end-1] = - recα(T, S, l, l, 3) / (α4 * recγ(T, S, l, l-1, 2))
    end
    S
end

function resizedata!(S::SphericalHarmonicsSpace, N)
    """ Resizes the data of S - that is, stores the Clenshaw (Recurrence)
    matrices up to degree N
    """

    N₀ = length(S.C)
    N ≤ N₀ - 2 && return S
    @show "begin resizedata! for SphericalHarmonicsSpace", N

    getAs!(S, N+1, N₀)
    @show "done As"
    # getBs!(S, N+1, N₀) # NOTE Bs are just square zero blocks
    # @show "done Bs"
    getCs!(S, N+1, N₀)
    @show "done Cs"
    getDTs!(S, N+1, N₀)
    @show "done DTs"
    S
end


function clenshawDTBmG(S::SphericalHarmonicsSpace, l::Int,
                        ξ::AbstractArray{T}, x::R, y::R, z::R;
                        clenshawalg::Bool=true) where {T,R}
    """ Returns the vector corresponding to:
        ξ * DTl * (Bl - Gl(x,y,z)) if clenshawalg
        DTl * (Bl - Gl(x,y,z)) * ξ if !clenshawalg
    where ξ is appropriatly sized vector.

    Note here that Bl's are assumed to be zero (as they are for SHs).
    """

    ret = - (S.DT[l+1][1] * x + S.DT[l+1][2] * y + S.DT[l+1][3] * z)
    if clenshawalg
        ξ * ret
    else
        ret * ξ
    end
end
function clenshawDTC(S::SphericalHarmonicsSpace, l::Int, ξ::AbstractArray;
                        clenshawalg::Bool=true)
    """ Returns vector corresponding to:
        ξ * DTl * Cl if clenshawalg
        DTl * Cl * ξ if !clenshawalg

    """

    ret = (S.DT[l+1][1] * S.C[l+1][1]
            + S.DT[l+1][2] * S.C[l+1][2]
            + S.DT[l+1][3] * S.C[l+1][3])
    if clenshawalg
        ξ * ret
    else
        ret * ξ
    end
end
function clenshaw(cfs::AbstractVector{T},
                    S::SphericalHarmonicsSpace,
                    x::R, y::R, z::R) where {T,R}
    """ Implements the Clenshaw algorithm to evaluate a function given by its
    expansion coeffs in the SH OP basis

    NOTE for now, we simply implement with the clenshaw mats as required.
    It could be made more efficient.
    """

    M = length(cfs)
    N = Int(sqrt(M)) - 1 # Degree
    resizedata!(S, N+1)
    f = PseudoBlockArray(cfs, [2n+1 for n=0:N])

    P0 = getdegzeropteval(T, S)
    if N == 0
        return f[1] * P0
    end
    ξ2 = transpose(view(f, Block(N+1)))
    ξ1 = transpose(view(f, Block(N))) - clenshawDTBmG(S, N-1, ξ2, x, y, z)
    for n = N-2:-1:0
        ξ = (transpose(view(f, Block(n+1)))
                - clenshawDTBmG(S, n, ξ1, x, y, z)
                - clenshawDTC(S, n+1, ξ2))
        ξ2 = copy(ξ1)
        ξ1 = copy(ξ)
    end
    (ξ1 * P0)[1]
end
clenshaw(cfs::AbstractVector, S::SphericalHarmonicsSpace, z) =
    clenshaw(cfs, S, z[1], z[2], z[3])
evaluate(cfs::AbstractVector, S::SphericalHarmonicsSpace, z) =
    clenshaw(cfs, S, z)
evaluate(cfs::AbstractVector, S::SphericalHarmonicsSpace, x, y, z) =
    clenshaw(cfs, S, x, y, z)



#===#
# Operator matrices

function laplacianoperator(S::SphericalHarmonicsSpace{<:Any, <:Any, T}, N::Int) where T
    """ Returns the deg N operator matrix for the diff operator Δ_s

    Returns as BandedBlockBandedMatrix
    """

    # ret = BandedBlockBandedMatrix(Zeros{B}((N+1)^2, (N+1)^2),
    #                                 [2l+1 for l = 1:N], [2l+1 for l = 1:N],
    #                                 (0, 0), (0, 0))
    ret = BandedBlockBandedMatrix(T(1)I, [2l+1 for l=0:N], [2l+1 for l=0:N],
                                    (0, 0), (0, 0))
    for l = 0:N
        view(ret, Block(l+1, l+1)) .*= - l * (l+1)
    end
    ret
end



# #-----
# # Testing
#
# x = 0.1
# y = 0.8
# z = sqrt(1 - x^2 - y^2)
# l,m = 3,-1
# p = sh_eval(l, m, x, y, z)
# p_actual = alphaVal(l,m) * (x - im*y) * ((z-1)^2 * 15/4 + (z-1) * 15/2 + 3)
# # p_actual = alphaVal(2,2) * (x + im*y)^2
# @test p ≈ p_actual
#
# N = 10
# f = 1:(N+1)^2
# fxyz = func_eval(f, x, y, z)
# p = sh_eval(N, x, y, z)
# fxyz_actual = 0.0
# for k = 0:N
#     fxyz_actual += vecdot(view(f, k^2+1:(k+1)^2), view(p, k^2+1:(k+1)^2))
# end
# fxyz_actual
# @test fxyz ≈ fxyz_actual
#
# N = 5
# f = 1:(N+1)^2
# fxyz = func_eval_jacobi(f)
#
# a = y*sh_eval(N,x,y,z)
# b = Jy(N)*sh_eval(N,x,y,z)
# (a - b)[1:N^2]
# @test a[1:N^2]≈b[1:N^2]
