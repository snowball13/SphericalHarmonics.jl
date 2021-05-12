# Vector Spherical Harmonics as basis functions on the tangent space of the unit
# sphere for a spectral method for solving PDEs

#=
NOTE

    VOPs: Ψ_l^m(x,y,z) := ∇Ylm(x,y,z) = ∂ϕ(Ylm) ϕ̲ + (1/sinϕ) ∂θ(Ylm) θ̲

for l ∈ ℕ₀, m = -l,...,l

where x = cosθ sinϕ, y = sinθ sinϕ, z = cosϕ; ρ(z) := sqrt(1-z^2) = sinϕ

=#

export SphericalHarmonicsTangentSpace

function SphericalHarmonicsTangentSpace(fam::SphericalHarmonicsFamily{B,T,<:Any}) where {B,T}
    SphericalHarmonicsTangentSpace{typeof(fam), B, T}(
        fam,
        Vector{Vector{BandedBlockBandedMatrix{B}}}(),
        Vector{Vector{BandedBlockBandedMatrix{B}}}(),
        Vector{Vector{BandedBlockBandedMatrix{B}}}(),
        Vector{Vector{SparseMatrixCSC{B}}}())
end

spacescompatible(A::SphericalHarmonicsTangentSpace, B::SphericalHarmonicsTangentSpace) = true

function gettangentspace(S::SphericalHarmonicsSpace{<:Any,B,T}) where {B,T}
    D = S.family
    if length(D.tangentspace) == 1
        D.tangentspace[1]
    elseif length(D.tangentspace) == 0
        resize!(D.tangentspace, 1)
        D.tangentspace[1] = SphericalHarmonicsTangentSpace(D)
    else
        error("tangentspace should be a vector of length 1 or 0")
    end
end

getSHspace(S::SphericalHarmonicsTangentSpace) = S.family()

function getorderonepteval(::Type{T}, S::SphericalHarmonicsTangentSpace,
                            x::R, y::R, z::R,
                            type::String="psi", m::Int=0) where {T,R}
    """ Returns the Ψ/Φ VSH OP evaluated at (x,y,z) for l = 1, m = m.

    im is the imaginary number i. m is simply 0 or ±1.
    """

    @assert in(type, ("psi", "phi")) "Invalid type of VSH asked for"
    @assert in(m, (0, 1, -1)) "Invalid order 1 VSH asked for (m should be 0, ±1)"

    if type == "psi"
        if m == 0
            ret = [-x*z; -y*z; -z^2 + 1]
        else
            ret = [1 - x^2 - m*im*x*y; -x*y + m*im*(1 - y^2); -z*(x + m*im*y)]
        end
    else
        if m == 0
            ret = [y; -x; 0]
        else
            ret = [-m*im*z; z; -y + m*im*x]
        end
    end
    ret * getnormalisingconstant(T, getSCspace(S), 1, m)
end
function orderonepteval(S::SphericalHarmonicsTangentSpace{<:Any, B, <:Any},
                        x::R, y::R, z::R,
                        cfs::AbstractArray{T}) where {B,R,T}
    """ Returns the result of the dot product of the cfs vector (length=6) with
    the order l=1 VSH OPs, yeilding a dim=3 vector.
    """

    ret = zeros(B, 3)
    types = ("psi", "phi")
    ind = 1
    for tp in types, m = -1:1
        ret += cfs[ind] * getorderonepteval(B, S, x, y, z)
        ind += 1
    end
    ret
end


#===#
# Recurrence coefficients/Jacobi matrix entries

function jacobiderivativecoeff(::Type{T}, S::SphericalHarmonicsTangentSpace,
                                l::Int, m::Int) where T
    """ Returns the coefficient d_{l,m} where
    dz(P^{(|m|,|m|)}_{l-|m|}(z)) = d_{l,m} P^{(|m|+1,|m|+1)}_{l-|m|-1}(z)

    """

    @assert l ≥ abs(m) "Invalid l, m given - l must be ≥ |m|"
    if l == abs(m)
        T(0)
    else
        (T(1) / 2) * (l + abs(m) + 1)
    end
end
jacobiderivativecoeff(::Type{T}, S::SphericalHarmonicsSpace, l::Int, m::Int) where T =
    jacobiderivativecoeff(T, gettangentspace(S), l, m)

function getrecÃ(::Type{T}, ST::SphericalHarmonicsTangentSpace,
                    l::Int, m::Int, j::Int) where T
    """ Returns the Ã_{l,m,j} value for j = 1,2,3,4 """

    SH = getSHspace(ST)
    ret = T(0)
    if j == 1
        ret += (m^2 - 1) * getrecα̃(T, SH, l-1, m-1, 4)
        ret -= l == m ? 0 : jacobiderivativecoeff(T, ST, l-1, m-1) * getrecγ̃(T, SH, l-1, m, 2)
        ret /= getnormalisingconstant(T, SH, l, m)^2
        if l != m
            ret += (jacobiderivativecoeff(T, ST, l, m)
                    * jacobiderivativecoeff(T, ST, l-1, m-1)
                    * getrecα̃(T, SH, l, m+1, 1)
                    / getnormalisingconstant(T, SH, l-1, m)^2)
        end
    elseif j == 2
        ret += m * (m+2) * getrecα̃(T, SH, l, m, 2)
        ret -= l == m ? 0 : jacobiderivativecoeff(T, ST, l, m) * getrecγ̃(T, SH, l, m+1, 1)
        ret /= getnormalisingconstant(T, SH, l-1, m+1)^2
        if l != m && l-1 != m+1
            ret += (jacobiderivativecoeff(T, ST, l, m)
                    * jacobiderivativecoeff(T, ST, l-1, m+1)
                    * getrecα̃(T, SH, l-1, m+2, 3)
                    / getnormalisingconstant(T, SH, l, m+1)^2)
        end
    elseif j == 3
        ret += (m^2 - 1) * getrecα̃(T, SH, l+1, m-1, 2)
        ret -= l+1 == m-1 ? 0 : jacobiderivativecoeff(T, ST, l+1, m-1) * getrecγ̃(T, SH, l+1, m, 1)
        ret /= getnormalisingconstant(T, SH, l, m)^2
        if l != m && l+1 != m-1
            ret += (jacobiderivativecoeff(T, ST, l, m)
                    * jacobiderivativecoeff(T, ST, l+1, m-1)
                    * getrecα̃(T, SH, l, m+1, 3)
                    / getnormalisingconstant(T, SH, l+1, m)^2)
        end
    elseif j == 4
        ret += m * (m+2) * getrecα̃(T, SH, l, m, 4)
        ret -= l == m ? 0 : jacobiderivativecoeff(T, ST, l, m) * getrecγ̃(T, SH, l, m+1, 2)
        ret /= getnormalisingconstant(T, SH, l+1, m+1)^2
        if l != m
            ret += (jacobiderivativecoeff(T, ST, l, m)
                    * jacobiderivativecoeff(T, ST, l+1, m+1)
                    * getrecα̃(T, SH, l+1, m+2, 1)
                    / getnormalisingconstant(T, SH, l, m+1)^2)
        end
    else
        error("Invalid Ã coeff being requested")
    end
    ret
end
function getrecΓ̃(::Type{T}, ST::SphericalHarmonicsTangentSpace,
                    l::Int, m::Int, j::Int) where T
    """ Returns the coeff, Γ̃_{l,m,j}, value for j = 1,2,3 """

    SH = getSHspace(ST)
    ret = T(0)
    mm = abs(m)
    if j == 1
        ret += (mm * (mm + 2) * getrecγ̃(T, SH, l, mm, 1)
                / getnormalisingconstant(T, SH, l-1, mm)^2)
        if l != mm && l-1 != mm
            ret += (jacobiderivativecoeff(T, ST, l, mm)
                    * jacobiderivativecoeff(T, ST, l-1, mm)
                    * getrecγ̃(T, SH, l, mm+1, 1)
                    / getnormalisingconstant(T, SH, l-1, mm+1)^2)
        end
    elseif j == 2
        ret += (mm * (mm + 2) * getrecγ̃(T, SH, l, mm, 2)
                / getnormalisingconstant(T, SH, l+1, mm)^2)
        if l != m && l+1 != mm
            ret += (jacobiderivativecoeff(T, ST, l, mm)
                    * jacobiderivativecoeff(T, ST, l+1, mm)
                    * getrecγ̃(T, SH, l, mm+1, 2)
                    / getnormalisingconstant(T, SH, l+1, mm+1)^2)
        end
    elseif j == 3
        ret += (-1)^m * im * m
    else
        error("Invalid Γ̃ coeff being requested")
    end
    ret
end
function recA(::Type{T}, S::SphericalHarmonicsTangentSpace, l::Int, m::Int, j::Int) where T
    """ Returns the mult by x coeff, A_{l,m,j}, value for j = 1,...,6 """

    @assert (j ≥ 1 && j ≤ 6 && l ≥ 0) "Invalid A coeff being requested"
    SH = getSHspace(S)
    ret = T(0)
    l == 0 && return ret

    if j == 1
        l-1 < abs(m-1) && return ret
        l-1 == 0 && m-1 == 0 && return ret
        if m > 0
            ret += getrecÃ(T, S, l, m, 1)
        else
            ret += getrecÃ(T, S, l, abs(m), 2)
        end
        ret *= (getnormalisingconstant(T, SH, l, m)
                * getnormalisingconstant(T, SH, l-1, m-1))
        ret /= 2 * l * (l-1)
    elseif j == 2
        l-1 < abs(m+1) && return ret
        l-1 == 0 && m+1 == 0 && return ret
        if m ≥ 0
            ret += getrecÃ(T, S, l, m, 2)
        else
            ret += getrecÃ(T, S, l, abs(m), 1)
        end
        ret *= (getnormalisingconstant(T, SH, l, m)
                * getnormalisingconstant(T, SH, l-1, m+1))
        ret /= 2 * l * (l-1)
    elseif j == 3
        l+1 < abs(m-1) && return ret
        if m > 0
            ret += getrecÃ(T, S, l, m, 3)
        else
            ret += getrecÃ(T, S, l, abs(m), 4)
        end
        ret *= (getnormalisingconstant(T, SH, l, m)
                * getnormalisingconstant(T, SH, l+1, m-1))
        ret /= 2 * (l+1) * (l+2)
    elseif j == 4
        l+1 < abs(m+1) && return ret
        if m ≥ 0
            ret += getrecÃ(T, S, l, m, 4)
        else
            ret += getrecÃ(T, S, l, abs(m), 3)
        end
        ret *= (getnormalisingconstant(T, SH, l, m)
                * getnormalisingconstant(T, SH, l+1, m+1))
        ret /= 2 * (l+1) * (l+2)
    elseif j == 5
        l < abs(m-1) && return ret
        if m > 0
            ret -= (im
                    * jacobiderivativecoeff(T, S, l, m-1)
                    * getnormalisingconstant(T, SH, l, m-1)
                    / (2 * getnormalisingconstant(T, SH, l, m)))
        else
            ret += ((-1)^m * im
                    * jacobiderivativecoeff(T, S, l, m)
                    * getnormalisingconstant(T, SH, l, m)
                    / (2 * getnormalisingconstant(T, SH, l, m-1)))
        end
        ret /= l * (l+1)
    elseif j == 6
        l < abs(m+1) && return ret
        if m ≥ 0
            ret -= (im
                    * jacobiderivativecoeff(T, S, l, m)
                    * getnormalisingconstant(T, SH, l, m)
                    / (2 * getnormalisingconstant(T, SH, l, m+1)))
        else
            ret += ((-1)^(m+1) * im
                    * jacobiderivativecoeff(T, S, l, abs(m)-1)
                    * getnormalisingconstant(T, SH, l, m+1)
                    / (2 * getnormalisingconstant(T, SH, l, m)))
        end
        ret /= l * (l+1)
    else
        error("Invalid A or B coeff being requested")
    end
    ret
end
function recB(::Type{T}, S::SphericalHarmonicsTangentSpace, l::Int, m::Int, j::Int) where T
    """ Returns the mult by y coeff, B_{l,m,j}, value for j = 1,2,3,4 """

    @assert (j ≥ 1 && j ≤ 6 && l ≥ 0) "Invalid B coeff being requested"
    (-1)^(j+1) * im * recA(T, S, l, m, j)
end
function recΓ(::Type{T}, S::SphericalHarmonicsTangentSpace, l::Int, m::Int, j::Int) where T
    """ Returns the mult by z coeff, Γ_{l,m,j}, value for j = 1,2,3 """

    @assert (j ≥ 1 && j ≤ 3 && l ≥ 0) "Invalid Γ coeff being requested"
    SH = getSHspace(S)
    ret = T(0)
    l == 0 && return ret
    j == 1 && l-1 < abs(m) && return ret
    j == 1 && l-1 == 0 && m == 0 && return ret

    ret += getrecΓ̃(T, S, l, m, j)
    if j == 1
        ret *= (getnormalisingconstant(T, SH, l, m)
                * getnormalisingconstant(T, SH, l-1, m)
                / (l * (l-1)))
    elseif j == 2
        ret *= (getnormalisingconstant(T, SH, l, m)
                * getnormalisingconstant(T, SH, l+1, m)
                / ((l+1) * (l+2)))
    else
        ret /= l * (l+1)
    end
    ret
end


#===#
# Function evaluation (clenshaw)

#=
NOTE
The Clenshaw matrices are stored by degree (and not by Fourier mode k).
This makes the Clenshaw algorithm much easier.
We will just need to reorder/take into account the fact that the coeffs are
stored by Fourier mode (and not degree) in the calculations.

OR since constructing/storing these takes a looong time, we do the clenshaw alg
when needed *not* using the clenshaw matrices.
=#

function getclenshawsubblockx(S::SphericalHarmonicsTangentSpace{<:Any, T, <:Any},
                                l::Int; subblock::String="A") where T
    """ Returns the Jacobi matrix subblock A_{x,l}, B_{x,l}, C_{x,l} """

    @assert subblock in ("A", "B", "C") "Invalid subblock given"
    @assert l ≥ 0 "Invalid l - should be non-negative integer"

    if subblock == "A"
        subblockmat = sparse(T(1)I, 2, 2)
        band = 1
        mat = BandedBlockBandedMatrix(
                Zeros{T}(2(2l+1), 2(2(l+band)+1)),
                [2 for m=-l:l], [2 for m=-(l+band):(l+band)],
                (0, 2), (0, 0))
        for i = 1:2l+1
            view(mat, Block(i, i)) .+= recA(T, S, l, -l+i-1, 3) * subblockmat
            view(mat, Block(i, i+2)) .+= recA(T, S, l, -l+i-1, 4) * subblockmat
        end
    elseif subblock == "B"
        subblockmat = sparse([0 T(1); -T(1) 0])
        band = 0
        mat = BandedBlockBandedMatrix(
                Zeros{T}(2(2l+1), 2(2(l+band)+1)),
                [2 for m=-l:l], [2 for m=-(l+band):(l+band)],
                (1, 1), (1, 1))
        for i = 1:2l
            view(mat, Block(i, i+1)) .+= recA(T, S, l, -l+i-1, 6) * subblockmat
            view(mat, Block(i+1, i)) .+= recA(T, S, l, -l+i, 5) * subblockmat
        end
    else
        l == 0 && error("l needs to be > 0 when Clenshaw mat C requested")
        subblockmat = sparse(T(1)I, 2, 2)
        band = -1
        mat = BandedBlockBandedMatrix(
                Zeros{T}(2(2l+1), 2(2(l+band)+1)),
                [2 for m=-l:l], [2 for m=-(l+band):(l+band)],
                (2, 0), (0, 0))
        for i = 1:2l-1
            view(mat, Block(i, i)) .+= recA(T, S, l, -l+i-1, 2) * subblockmat
            view(mat, Block(i+2, i)) .+= recA(T, S, l, -l+i+1, 1) * subblockmat
        end
    end
    mat
end
function getclenshawsubblocky(S::SphericalHarmonicsTangentSpace{<:Any, T, <:Any},
                                l::Int; subblock::String="A") where T
    """ Returns the Jacobi matrix subblock A_{y,l}, B_{y,l}, C_{y,l} """

    @assert subblock in ("A", "B", "C") "Invalid subblock given"
    @assert l ≥ 0 "Invalid l - should be non-negative integer"

    if subblock == "A"
        subblockmat = sparse(T(1)I, 2, 2)
        band = 1
        mat = BandedBlockBandedMatrix(
                Zeros{T}(2(2l+1), 2(2(l+band)+1)),
                [2 for m=-l:l], [2 for m=-(l+band):(l+band)],
                (0, 2), (0, 0))
        for i = 1:2l+1
            view(mat, Block(i, i)) .+= recB(T, S, l, -l+i-1, 3) * subblockmat
            view(mat, Block(i, i+2)) .+= recB(T, S, l, -l+i-1, 4) * subblockmat
        end
    elseif subblock == "B"
        subblockmat = sparse([0 T(1); -T(1) 0])
        band = 0
        mat = BandedBlockBandedMatrix(
                Zeros{T}(2(2l+1), 2(2(l+band)+1)),
                [2 for m=-l:l], [2 for m=-(l+band):(l+band)],
                (1, 1), (1, 1))
        for i = 1:2l
            view(mat, Block(i, i+1)) .+= recB(T, S, l, -l+i-1, 6) * subblockmat
            view(mat, Block(i+1, i)) .+= recB(T, S, l, -l+i, 5) * subblockmat
        end
    else
        l == 0 && error("l needs to be > 0 when Clenshaw mat C requested")
        subblockmat = sparse(T(1)I, 2, 2)
        band = -1
        mat = BandedBlockBandedMatrix(
                Zeros{T}(2(2l+1), 2(2(l+band)+1)),
                [2 for m=-l:l], [2 for m=-(l+band):(l+band)],
                (2, 0), (0, 0))
        for i = 1:2l-1
            view(mat, Block(i, i)) .+= recB(T, S, l, -l+i-1, 2) * subblockmat
            view(mat, Block(i+2, i)) .+= recB(T, S, l, -l+i-1, 1) * subblockmat
        end
    end
    mat
end
function getclenshawsubblockz(S::SphericalHarmonicsTangentSpace{<:Any, T, <:Any},
                                l::Int; subblock::String="A") where T
    """ Returns the Jacobi matrix subblock A_{z,l}, B_{z,l}, C_{z,l} """

    @assert subblock in ("A", "B", "C") "Invalid subblock given"
    @assert l ≥ 0 "Invalid l - should be non-negative integer"

    if subblock == "A"
        subblockmat = sparse(T(1)I, 2, 2)
        band = 1
        mat = BandedBlockBandedMatrix(
                Zeros{T}(2(2l+1), 2(2(l+band)+1)),
                [2 for m=-l:l], [2 for m=-(l+band):(l+band)],
                (-1, 1), (0, 0))
        for i = 1:2l+1
            view(mat, Block(i, i+1)) .+= recΓ(T, S, l, -l+i-1, 2) * subblockmat
        end
    elseif subblock == "B"
        subblockmat = sparse([0 T(1); -T(1) 0])
        band = 0
        mat = BandedBlockBandedMatrix(
                Zeros{T}(2(2l+1), 2(2(l+band)+1)),
                [2 for m=-l:l], [2 for m=-(l+band):(l+band)],
                (0, 0), (1, 1))
        for i = 1:2l+1
            view(mat, Block(i, i)) .+= recΓ(T, S, l, -l+i-1, 3) * subblockmat
        end
    else
        l == 0 && error("l needs to be > 0 when Clenshaw mat C requested")
        subblockmat = sparse(T(1)I, 2, 2)
        band = -1
        mat = BandedBlockBandedMatrix(
                Zeros{T}(2(2l+1), 2(2(l+band)+1)),
                [2 for m=-l:l], [2 for m=-(l+band):(l+band)],
                (1, -1), (0, 0))
        for i = 1:2l-1
            view(mat, Block(i+1, i)) .+= recΓ(T, S, l, -l+i, 1) * subblockmat
        end
    end
    mat
end

# NOTE Each of these """ Computes and stores the Jacobi matrix blocks up to deg N """
function getBs!(S::SphericalHarmonicsTangentSpace{<:Any, T, <:Any}, N, N₀) where T
    m = N₀
    resize!(S.B, N + 1)
    subblock = "B"
    for n = N:-1:m
        S.B[n+1] = Vector{BandedBlockBandedMatrix{T}}(undef, 3)
        resize!(S.B[n+1], 3)
        S.B[n+1][1] = getclenshawsubblockx(S, n; subblock=subblock)
        S.B[n+1][2] = getclenshawsubblocky(S, n; subblock=subblock)
        S.B[n+1][3] = getclenshawsubblockz(S, n; subblock=subblock)
    end
    S
end
function getCs!(S::SphericalHarmonicsTangentSpace{<:Any, T, <:Any}, N, N₀) where T
    m = N₀
    resize!(S.C, N + 1)
    subblock = "C"
    if N₀ == 0
        m += 1 # C_0 does not exist
    end
    for n = N:-1:m
        S.C[n+1] = Vector{BandedBlockBandedMatrix{T}}(undef, 3)
        resize!(S.C[n+1], 3)
        S.C[n+1][1] = getclenshawsubblockx(S, n; subblock=subblock)
        S.C[n+1][2] = getclenshawsubblocky(S, n; subblock=subblock)
        S.C[n+1][3] = getclenshawsubblockz(S, n; subblock=subblock)
    end
    S
end
function getAs!(S::SphericalHarmonicsTangentSpace{<:Any, T, <:Any}, N, N₀) where T
    m = N₀
    resize!(S.A, N + 1)
    subblock = "A"
    for n = N:-1:m
        S.A[n+1] = Vector{BandedBlockBandedMatrix{T}}(undef, 3)
        resize!(S.A[n+1], 3)
        S.A[n+1][1] = getclenshawsubblockx(S, n; subblock=subblock)
        S.A[n+1][2] = getclenshawsubblocky(S, n; subblock=subblock)
        S.A[n+1][3] = getclenshawsubblockz(S, n; subblock=subblock)
    end
    S
end

function getDTs!(S::SphericalHarmonicsTangentSpace{<:Any, T, <:Any}, N, N₀) where T
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
        A3, A4 = recA(T, S, l, l, 3), recA(T, S, l, l, 4)
        B3, B4 = recB(T, S, l, l, 3), recB(T, S, l, l, 4)
        denom = A3 * B4 - A4 * B3

        S.DT[l+1][1] = spzeros(T, 2(2l+3), 2(2l+1))
        Dx = S.DT[l+1][1]
        Dx[1,1] = B4 / denom; Dx[2,2] = B4 / denom
        Dx[5,1] = -B3 / denom; Dx[6,2] = -B3 / denom

        S.DT[l+1][2] = spzeros(T, 2(2l+3), 2(2l+1))
        Dy = S.DT[l+1][2]
        Dy[1,1] = -A4 / denom; Dx[2,2] = -A4 / denom
        Dy[5,1] = A3 / denom; Dx[6,2] = A3 / denom

        S.DT[l+1][3] = spzeros(T, 2(2l+3), 2(2l+1))
        Dz = S.DT[l+1][3]
        ent = 1 / recΓ(T, S, l, l, 2)
        Dz[3,1] = ent; Dz[4,2] = ent

        previousN += 1
    end
    for l = N:-1:previousN
        S.DT[l+1] = Vector{SparseMatrixCSC{T}}(undef, 3)
        resize!(S.DT[l+1], 3)

        # Define
        S.DT[l+1][1] = spzeros(T, 2(2l+3), 2(2l+1))
        Dx = S.DT[l+1][1]
        A3, A4 = recA(T, S, l, -l, 3), recA(T, S, l, l, 4)
        Dx[1, 1] = 1 / A3; Dx[2, 2] = 1 / A3
        Dx[end-1, end-1] = 1 / A4; Dx[end, end] = 1 / A4

        S.DT[l+1][2] = spzeros(T, 2(2l+3), 2(2l+1))

        S.DT[l+1][3] = spzeros(T, 2(2l+3), 2(2l+1))
        Dz = S.DT[l+1][3]
        ent = - recA(T, S, l, -l, 4) / (A3 * recΓ(T, S, l, -l+1, 2))
        Dz[1, 3] = ent; Dz[2, 4] = ent
        offset = 2
        ind = 1
        for m = -l:l
            c = recΓ(T, S, l, m, 2)
            for i = 1:2
                Dz[offset+ind, ind] = 1 / c
                ind += 1
            end
        end
        ent = - recA(T, S, l, l, 3) / (A4 * recΓ(T, S, l, l-1, 2))
        Dz[end-1, end-5] = ent; Dz[end, end-4] = ent
    end
    S
end

function resizedata!(S::SphericalHarmonicsTangentSpace, N)
    """ Resizes the data of S - that is, stores the Clenshaw (Recurrence)
    matrices up to degree N
    """

    N₀ = length(S.C)
    N ≤ N₀ - 2 && return S
    @show "begin resizedata! for SphericalHarmonicsTangentSpace", N

    getAs!(S, N+1, N₀)
    @show "done As"
    getBs!(S, N+1, N₀)
    @show "done Bs"
    getCs!(S, N+1, N₀)
    @show "done Cs"
    getDTs!(S, N+1, N₀)
    @show "done DTs"
    S
end


function clenshawDTBmG(S::SphericalHarmonicsTangentSpace{<:Any, T, <:Any},
                        l::Int, ξ::AbstractArray{R}, x::R, y::R, z::R) where {T,R}
    """ Returns Vector corresponding to ξ * DlT * (Bl - Gl(x,y,z)) """

    - ξ * (S.DT[l+1][1] * x + S.DT[l+1][2] * y + S.DT[l+1][3] * z)
end
function clenshawDTC(S::SphericalHarmonicsTangentSpace{<:Any, T, <:Any}, l::Int,
                        ξ::AbstractArray{R}) where {T,R}
    """ Returns Vector corresponding to ξ * DlT * Cl """

    ξ * (S.DT[l+1][1] * S.C[l+1][1]
            + S.DT[l+1][2] * S.C[l+1][2]
            + S.DT[l+1][3] * S.C[l+1][3])
end
function clenshaw(cfs::AbstractVector{T},
                    S::SphericalHarmonicsTangentSpace{<:Any, B, <:Any},
                    x::R, y::R, z::R) where {T,B,R}
    """ Implements the Clenshaw algorithm to evaluate a function given by its
    expansion coeffs in the SH OP basis

    NOTE for now, we simply implement with the clenshaw mats as required.
    It could be made more efficient.
    """

    M = length(cfs)
    N = Int(sqrt(Int(M/2))) - 1 # Degree
    resizedata!(S, N+1)
    f = PseudoBlockArray(cfs, [2(2l+1) for l=0:N])

    if N == 0
        return zeros(B, 3)
    elseif N == 1
        return orderonepteval(S, x, y, z, view(f, Block(N+1)))
    end

    ξ2 = view(f, Block(N+1))'
    ξ1 = view(f, Block(N))' - clenshawDTBmG(S, N-1, ξ2, x, y, z)
    for n = N-2:-1:1
        ξ = (view(f, Block(n+1))'
                - clenshawDTBmG(S, n, ξ1, x, y, z)
                - clenshawDTC(S, n+1, ξ2))
        ξ2 = copy(ξ1)
        ξ1 = copy(ξ)
    end
    orderonepteval(S, x, y, z, ξ1)
end
clenshaw(cfs::AbstractVector, S::SphericalHarmonicsTangentSpace, z) =
    clenshaw(cfs, S, z[1], z[2], z[3])
evaluate(cfs::AbstractVector, S::SphericalHarmonicsTangentSpace, z) =
    clenshaw(cfs, S, z)
evaluate(cfs::AbstractVector, S::SphericalHarmonicsTangentSpace, x, y, z) =
    clenshaw(cfs, S, x, y, z)


#===#
# Operator matrices

function gradientoperator(S::SphericalHarmonicsTangentSpace{<:Any, B, T},
                            N::Int; small::Bool=false) where {B,T}
    """ Returns the deg N operator matrix for the diff operator ∇_s (grad)

    Acting on ℙ (SH OP vec) coeffs, resulting in coeffs in 𝕋^Ψ if small=true.
    Else acting on extended ℙ (SH OP vec) coeffs vec, resulting in coeffs in 𝕋
        (if small!=true).

    Returns as BandedBlockBandedMatrix.
    """

    if small
        BandedBlockBandedMatrix(B(1)I, [2l+1 for l=0:N], [2l+1 for l=0:N],
                                (0, 0), (0, 0))
    else
        ret = BandedBlockBandedMatrix(B(0)I, [2(2l+1) for l=0:N], [2(2l+1) for l=0:N],
                                        (0, 0), (0, 0))
        for l = 0:N
            view(ret, Block(l+1, l+1)) .= Diagonal(vcat([[B(1); B(0)] for m = -l:l]...))
        end
        ret
    end
end
gradientoperator(S::SphericalHarmonicsSpace, N::Int; small::Bool=false) =
    gradientoperator(gettangentspace(S), N; small=small)

function divergenceoperator(S::SphericalHarmonicsTangentSpace{<:Any, B, T},
                            N::Int; small::Bool=false) where {B,T}
    """ Returns the deg N operator matrix for the diff operator ∇_s⋅ (divergence)

    Acting on 𝕋^Ψ (VSH OP vec) coeffs, resulting in coeffs in ℙ if small=true.
    Else acting on extended 𝕋 (VSH OP vec) coeffs vec, resulting in coeffs in
        extended ℙ (if small!=true).

    Returns as BandedBlockBandedMatrix.
    """

    if small
        laplacianoperator(getSHspace(S), N)
    else
        ret = BandedBlockBandedMatrix(B(0)I, [2(2l+1) for l=0:N], [2(2l+1) for l=0:N],
                                        (0, 0), (0, 0))
        for l = 0:N
            view(ret, Block(l+1, l+1)) .= Diagonal(vcat([[- l * (l+1); T(0)] for m = -l:l]...))
        end
        ret
    end
end

function unitcrossproductoperator(S::SphericalHarmonicsTangentSpace{<:Any, B, T},
                            N::Int) where {B,T}
    """ Returns the (square) operator matrix representing r̂ × _ (cross product
    by the unit normal vector).

    Returns as BandedBlockBandedMatrix
    """

    ret = BandedBlockBandedMatrix(B(0)I, [2(2l+1) for l=0:N], [2(2l+1) for l=0:N],
                                    (0, 0), (1, 1))
    ind = 1
    for j = 1:(N+1)^2
        view(ret, ind, ind+1) .= B(1)
        view(ret, ind+1, ind) .= -B(1)
        ind += 2
    end
    ret
end


#===#
# Jacobi matrices (square or otherwise)

function jacobiz(S::SphericalHarmonicsTangentSpace{<:Any, B, T},
                    N::Int) where {B,T}
    """ Returns (square) jacobi matrix for mult by z of the tangent space VSH
        basis.

        Matrix is transposed so as to act on the coeffs vec directly.
    """

    resizedata!(S, N)
    matind = 3 # repesents x, y or z
    J = BandedBlockBandedMatrix(Zeros{B}(2(N+1)^2, 2(N+1)^2),
                                [2(2l+1) for l=0:N], [2(2l+1) for l=0:N],
                                (1, 1), (2, 2))
    # Assign each block
    l = 0
    view(J, Block(l+1, l+1)) .= transpose(S.B[l+1][matind])
    for l = 1:N
        view(J, Block(l+1, l)) .= transpose(S.A[l][matind])
        view(J, Block(l+1, l+1)) .= transpose(S.B[l+1][matind])
        view(J, Block(l, l+1)) .= transpose(S.C[l+1][matind])
    end
    J
end



#========#


#
# #====#
# # Testing
#
#
# tol = 1e-10
#
#
#
# N = 10
# Dx = grad_sh(N, 1)
# Dy = grad_sh(N, 2)
# Dz = grad_sh(N, 3)
# # I would expect this to just be diagonal (like the Laplacian)
# D2 = Dx^2 + Dy^2 + Dz^2
# Lap = laplacian_sh(N)
# # D2 then matches Lap (ignoring the last 2N+1 rows/cols, since the matrices for
# # Dx are not true representations for derivatives at these entries)
# B = abs.(D2 - Lap)[1:end-(2N+1), 1:end-(2N+1)]
# @test count(i->i>tol, B) == 0
#
# DPerpx = grad_perp_sh(N, 1)
# DPerpy = grad_perp_sh(N, 2)
# DPerpz = grad_perp_sh(N, 3)
# D2Perp = DPerpx^2 + DPerpy^2 + DPerpz^2
# # D2 then matches Lap (ignoring the last 2N+1 rows/cols, since the matrices for
# # Dx are not true representations for derivatives at these entries)
# B = abs.(D2Perp - Lap)[1:end-(2N+1), 1:end-(2N+1)]
# @test count(i->i>tol, B) == 0
#
# ######
#
# x, y = 0.5, 0.1
# z = sqrt(1 - x^2 - y^2)
# Y = sh_eval(N, x, y, z)
# DxY = Dx*Y
# DPerpxY = DPerpx*Y
# for l = 1:6
#     for m = -l:l
#         @test abs(x*DxY[l^2+l+1+m] - (coeff_a(l, m)*DxY[(l+1)^2+l+1+1+m+1]
#                                         + coeff_b(l, m)*DxY[(l-1)^2+l-1+1+m+1]
#                                         + coeff_d(l, m)*DxY[(l+1)^2+l+1+1+m-1]
#                                         + coeff_e(l, m)*DxY[max(1,(l-1)^2+l-1+1+m-1)]
#                                         + coeff_h(l, m)*DPerpxY[l^2+l+1+m+1]
#                                         + coeff_j(l, m)*DPerpxY[l^2+l+1+m-1])
#             ) < tol
#         @test abs(y*DxY[l^2+l+1+m] - (- im * coeff_a(l, m)*DxY[(l+1)^2+l+1+1+m+1]
#                                         - im * coeff_b(l, m)*DxY[(l-1)^2+l-1+1+m+1]
#                                         + im * coeff_d(l, m)*DxY[(l+1)^2+l+1+1+m-1]
#                                         + im * coeff_e(l, m)*DxY[max(1,(l-1)^2+l-1+1+m-1)]
#                                         - im * coeff_h(l, m)*DPerpxY[l^2+l+1+m+1]
#                                         + im * coeff_j(l, m)*DPerpxY[l^2+l+1+m-1])
#             ) < tol
#         @test abs(z*DxY[l^2+l+1+m] - (coeff_f(l, m)*DxY[(l+1)^2+l+1+1+m]
#                                         + coeff_g(l, m)*DxY[max(1, (l-1)^2+l-1+1+m)]
#                                         + coeff_k(l, m)*DPerpxY[l^2+l+1+m])
#             ) < tol
#
#         @test abs(x*DPerpxY[l^2+l+1+m] - (perp_coeff_a(l, m)*DPerpxY[(l+1)^2+l+1+1+m+1]
#                                         + perp_coeff_b(l, m)*DPerpxY[(l-1)^2+l-1+1+m+1]
#                                         + perp_coeff_d(l, m)*DPerpxY[(l+1)^2+l+1+1+m-1]
#                                         + perp_coeff_e(l, m)*DPerpxY[max(1,(l-1)^2+l-1+1+m-1)]
#                                         + perp_coeff_h(l, m)*DxY[l^2+l+1+m+1]
#                                         + perp_coeff_j(l, m)*DxY[l^2+l+1+m-1])
#             ) < tol
#         @test abs(y*DPerpxY[l^2+l+1+m] - (- im * perp_coeff_a(l, m)*DPerpxY[(l+1)^2+l+1+1+m+1]
#                                             - im * perp_coeff_b(l, m)*DPerpxY[(l-1)^2+l-1+1+m+1]
#                                             + im * perp_coeff_d(l, m)*DPerpxY[(l+1)^2+l+1+1+m-1]
#                                             + im * perp_coeff_e(l, m)*DPerpxY[max(1,(l-1)^2+l-1+1+m-1)]
#                                             - im * perp_coeff_h(l, m)*DxY[l^2+l+1+m+1]
#                                             + im * perp_coeff_j(l, m)*DxY[l^2+l+1+m-1])
#             ) < tol
#         @test abs(z*DPerpxY[l^2+l+1+m] - (perp_coeff_f(l, m)*DPerpxY[(l+1)^2+l+1+1+m]
#                                             + perp_coeff_g(l, m)*DPerpxY[max(1, (l-1)^2+l-1+1+m)]
#                                             + perp_coeff_k(l, m)*DxY[l^2+l+1+m])
#             ) < tol
#     end
# end
#
#
# #####
#
#
# # a = x*tangent_basis_eval(N,x,y,z)
# # b = grad_Jx(N)*tangent_basis_eval(N,x,y,z)
# # c = abs.(a[1:6N^2] - b[1:6N^2])
# # @test count(i->i>tol, c) == 0
# # a = y*tangent_basis_eval(N,x,y,z)
# # b = grad_Jy(N)*tangent_basis_eval(N,x,y,z)
# # c = abs.(a[1:6N^2] - b[1:6N^2])
# # @test count(i->i>tol, c) == 0
# # a = z*tangent_basis_eval(N,x,y,z)
# # b = grad_Jz(N)*tangent_basis_eval(N,x,y,z)
# # c = abs.(a[1:6N^2] - b[1:6N^2])
# # @test count(i->i>tol, c) == 0
#
#
# #####
#
#
# N = 5
# f = 2*ones(2(N+1)^2)
# ∇P1 = tangent_basis_eval(N,x,y,z)
# feval = tangent_func_eval(f,x,y,z)
# feval_actual = zeros(3)
# for i=1:length(f)
#         feval_actual += f[i] * view(∇P1, Block(i))
# end
# @test feval_actual ≈ feval
