#=

Functions for determining the gradient and perpendicular gradient of spherical
harmonics (OPs) on the unit sphere.

The maths behind this stems from the gradient of a spherical harmonic (SH)
is the sum of two vector spherical harmoncs (VSH). A VSH can be written as a
sum of SHs with weights as Clebsch-Gorden coeffs times a vector (an
eigenvector of S3).
http://scipp.ucsc.edu/~haber/ph216/clebsch.pdf

=#

# include("SphericalHarmonicsScalar.jl")


let

    global function betaVal(l, r)
        if r < 0
            return (l+1)*sqrt(l/(2l+1))
        elseif r > 0
            return l*sqrt((l+1)/(2l+1))
        else
            return im*sqrt(l*(l+1))
        end
    end

    #=
    Function that returns the Clebsch-Gordan coefficient
        <L, M-ms; 1, ms | J, M>
    =#
    global function clebsch_gordan_coeff(J, M, L, ms)
        out = 0.0
        if ms == 1
            if J == L+1
                out = sqrt((L+M)*(L+M+1) / ((2L+1)*(2L+2)))
            elseif J == L
                out = - sqrt((L+M)*(L-M+1) / (2L*(L+1)))
            elseif J == L-1
                out = sqrt((L-M)*(L-M+1) / (2L*(2L+1)))
            end
        elseif ms == 0
            if J == L+1
                out = sqrt((L-M+1)*(L+M+1) / ((L+1)*(2L+1)))
            elseif J == L
                out = M / sqrt(L*(L+1))
            elseif J == L-1
                out = - sqrt((L-M)*(L+M) / (L*(2L+1)))
            end
        elseif ms == -1
            if J == L+1
                out = sqrt((L-M)*(L-M+1) / ((2L+1)*(2L+2)))
            elseif J == L
                out = sqrt((L-M)*(L+M+1) / (2L*(L+1)))
            elseif J == L-1
                out = sqrt((L+M)*(L+M+1) / (2L*(2L+1)))
            end
        end
        return out
    end

    function spin_func(r)
        if r == 1 || r == -1
            return [-r, -im, 0] / sqrt(2)
        elseif r == 0
            return [0, 0, 1]
        else
            return [0, 0, 0]
        end
    end


    #=
    Coefficients from the relations of multiplication by x,y,z etc on
    grad(Y_l^m) and grad⟂(Y_l^m). Used as Entries in the Jx, Jy, Jz jacobi
    matrices
    =#

    global function coeff_a(l, m)
        ms = 0
        a = coeff_A(l+1, m-ms) * betaVal(l, 1) * clebsch_gordan_coeff(l, m, l+1, ms)
        a /= (betaVal(l+1, 1) * clebsch_gordan_coeff(l+1, m+1, l+2, ms))
        return a
    end

    global function coeff_b(l, m)
        b = 0
        diff = l - m
        if l > 1 && diff > 1
            ms = -1
            if diff == 2
                ms = 1
            elseif diff == 3
                ms = 0
            end
            b = coeff_B(l-1, m-ms) * betaVal(l, -1) * clebsch_gordan_coeff(l, m, l-1, ms)
            b /= (betaVal(l-1, -1) * clebsch_gordan_coeff(l-1, m+1, l-2, ms))
        end
        return b
    end

    global function coeff_d(l, m)
        ms = 0
        d = coeff_D(l+1, m-ms) * betaVal(l, 1) * clebsch_gordan_coeff(l, m, l+1, ms)
        d /= (betaVal(l+1, 1) * clebsch_gordan_coeff(l+1, m-1, l+2, ms))
        return d
    end

    global function coeff_e(l, m)
        e = 0
        diff = l + m
        if l > 1 && diff > 1
            ms = 1
            if diff == 2
                ms = -1
            elseif diff == 3
                ms = 0
            end
            e = coeff_E(l-1, m-ms) * betaVal(l, -1) * clebsch_gordan_coeff(l, m, l-1, ms)
            e /= (betaVal(l-1, -1) * clebsch_gordan_coeff(l-1, m-1, l-2, ms))
        end
        return e
    end

    global function coeff_f(l, m)
        ms = 0
        f = coeff_F(l+1, m-ms) * betaVal(l, 1) * clebsch_gordan_coeff(l, m, l+1, ms)
        f /= (betaVal(l+1, 1) * clebsch_gordan_coeff(l+1, m, l+2, ms))
        return f
    end

    global function coeff_g(l, m)
        g = 0
        if l > 1 && l != abs(m)
            diff = l - m
            ms = -1
            if diff == 1
                ms = 1
            elseif diff == 2
                ms = 0
            end
            g = coeff_G(l-1, m-ms) * betaVal(l, -1) * clebsch_gordan_coeff(l, m, l-1, ms)
            g /= (betaVal(l-1, -1) * clebsch_gordan_coeff(l-1, m, l-2, ms))
        end
        return g
    end

    global function coeff_h(l, m)
        h = 0
        if l != m
            ms = -1
            if m == 0 || (m > 0 && l - m == 1)
                ms = 0
            end
            h = coeff_A(l-1,m-ms)*betaVal(l,-1)*clebsch_gordan_coeff(l,m,l-1,ms)
            h += coeff_B(l+1,m-ms)*betaVal(l,1)*clebsch_gordan_coeff(l,m,l+1,ms)
            h -= coeff_a(l,m)*betaVal(l+1,-1)*clebsch_gordan_coeff(l+1,m+1,l,ms)
            h -= coeff_b(l,m)*betaVal(l-1,1)*clebsch_gordan_coeff(l-1,m+1,l,ms)
            h /= betaVal(l,0)*clebsch_gordan_coeff(l,m+1,l,ms)
        end
        return h
    end

    global function coeff_j(l, m)
        j = 0
        if l != -m
            ms = 1
            if m == 0 || (m < 0 && l + m == 1)
                ms = 0
            end
            j = coeff_D(l-1,m-ms)*betaVal(l,-1)*clebsch_gordan_coeff(l,m,l-1,ms)
            j += coeff_E(l+1,m-ms)*betaVal(l,1)*clebsch_gordan_coeff(l,m,l+1,ms)
            j -= coeff_d(l,m)*betaVal(l+1,-1)*clebsch_gordan_coeff(l+1,m-1,l,ms)
            j -= coeff_e(l,m)*betaVal(l-1,1)*clebsch_gordan_coeff(l-1,m-1,l,ms)
            j /= betaVal(l,0)*clebsch_gordan_coeff(l,m-1,l,ms)
        end
        return j
    end

    global function coeff_k(l, m)
        k = 0
        if m != 0
            ms = sign(m)
            k = coeff_F(l-1,m-ms)*betaVal(l,-1)*clebsch_gordan_coeff(l,m,l-1,ms)
            k += coeff_G(l+1,m-ms)*betaVal(l,1)*clebsch_gordan_coeff(l,m,l+1,ms)
            k -= coeff_f(l,m)*betaVal(l+1,-1)*clebsch_gordan_coeff(l+1,m,l,ms)
            k -= coeff_g(l,m)*betaVal(l-1,1)*clebsch_gordan_coeff(l-1,m,l,ms)
            k /= betaVal(l,0)*clebsch_gordan_coeff(l,m,l,ms)
        end
        return k
    end

    global function perp_coeff_a(l, m)
        return coeff_a(l, m)'
    end

    global function perp_coeff_b(l, m)
        return coeff_b(l, m)'
    end

    global function perp_coeff_d(l, m)
        return coeff_d(l, m)'
    end

    global function perp_coeff_e(l, m)
        return coeff_e(l, m)'
    end

    global function perp_coeff_f(l, m)
        return coeff_f(l, m)'
    end

    global function perp_coeff_g(l, m)
        return coeff_g(l, m)'
    end

    global function perp_coeff_h(l, m)
        return coeff_h(l, m)'
    end

    global function perp_coeff_j(l, m)
        return coeff_j(l, m)'
    end

    global function perp_coeff_k(l, m)
        return coeff_k(l, m)'
    end


    #=
    Sub-Matrices for the Jacobi operator matrices
    =#
    function get_Ax_submatrices(n)
        dim = 2(2n+1)
        leftdiag = zeros(Complex128, dim)
        rightdiag = copy(leftdiag)
        # Gather non-zero entries
        d = a = im
        index = 1
        for k = -n:n
            a = coeff_a(n, k)
            d = coeff_d(n, k)
            view(leftdiag, index:index+1) .= d, d'
            view(rightdiag, index:index+1) .= a, a'
            index += 2
        end
        zerosMatrix = zeros(dim,4)
        return [Diagonal(leftdiag) zerosMatrix], [zerosMatrix Diagonal(rightdiag)]
    end

    global function grad_jacobi_Ax(n)
        leftmat, rightmat = get_Ax_submatrices(n)
        # Assemble full sub matrix
        return leftmat + rightmat
    end

    global function grad_jacobi_Ay(n)
        leftmat, rightmat = get_Ax_submatrices(n)
        # Assemble full sub matrix
        return im*(leftmat - rightmat)
    end

    global function grad_jacobi_Az(n)
        dim = 2(2n+1)
        d = zeros(Complex128, dim)
        # Gather non-zero entries
        f = im
        index = 1
        for k = -n:n
            f = coeff_f(n, k)
            view(d, index:index+1) .= f, f'
            index += 2
        end
        # Assemble full sub matrix
        zerosMatrix = zeros(dim, 2)
        return [zerosMatrix Diagonal(d) zerosMatrix]
    end

    function get_Bx_submatrices(n)
        dim = 2(2n)
        u = zeros(Complex128, dim-1)
        uperp = zeros(Complex128, dim+1)
        l = copy(uperp)
        lperp = copy(u)
        # Gather non-zero entries
        j = im
        h = coeff_h(n, -n)
        view(u, 1) .= h
        view(uperp, 2) .= h'
        index = 1
        for k = -n+1:n-1
            j = coeff_j(n, k)
            h = coeff_h(n, k)
            view(lperp, index) .= j'
            view(l, index+1) .= j
            view(u, index+2) .= h
            view(uperp, index+3) .= h'
            index += 2
        end
        j = coeff_j(n, n)
        view(lperp, index) .= j'
        view(l, index+1) .= j
        return diagm(u, 3), diagm(uperp, 1), diagm(l, -1), diagm(lperp, -3)
    end

    global function grad_jacobi_Bx(n)
        if n == 0
            return zeros(Complex128, 2, 2)
        end
        u, uperp, l, lperp = get_Bx_submatrices(n)
        # Assemble full sub matrix
        return sparse(u + uperp + l + lperp)
    end

    global function grad_jacobi_By(n)
        if n == 0
            return zeros(Complex128, 2,2)
        end
        u, uperp, l, lperp = get_Bx_submatrices(n)
        # Assemble full sub matrix
        return sparse(im*(-u - uperp + l + lperp))
    end

    global function grad_jacobi_Bz(n)
        dim = 2(2n+1)
        superdiag = zeros(Complex128, dim-1)
        subdiag = copy(superdiag)
        # Gather non-zero entries
        index = 1
        k = im
        for r = -n:n
            k = coeff_k(n, r)
            view(superdiag, index) .= k
            view(subdiag, index) .= k'
            index += 2
        end
        # Assemble full sub matrix
        return Tridiagonal(subdiag, zeros(dim), superdiag)
    end

    function get_Cx_submatrices(n)
        dim = 2(2n-1)
        upperdiag = zeros(Complex128, dim)
        lowerdiag = copy(upperdiag)
        # Gather non-zero entries
        index = 1
        b = e = im
        for k = -n:n-2
            b = coeff_b(n,k)
            e = coeff_e(n,k+2)
            view(upperdiag, index:index+1) .= b, b'
            view(lowerdiag, index:index+1) .= e, e'
            index += 2
        end
        zerosMatrix = zeros(4,dim)
        return [Diagonal(upperdiag); zerosMatrix], [zerosMatrix; Diagonal(lowerdiag)]
    end

    global function grad_jacobi_Cx(n)
        uppermat, lowermat = get_Cx_submatrices(n)
        # Assemble full sub matrix, exploiting the symmetry of the system
        return uppermat + lowermat
    end

    global function grad_jacobi_Cy(n)
        uppermat, lowermat = get_Cx_submatrices(n)
        # Assemble full sub matrix, exploiting the symmetry of the system
        return im*(-uppermat + lowermat)
    end

    global function grad_jacobi_Cz(n)
        dim = 2(2n-1)
        d = zeros(Complex128, dim)
        # Gather non-zero entries
        index = 1
        g = im
        for k = -n+1:n-1
            g = coeff_g(n,k)
            view(d, index:index+1) .= g, g'
            index += 2
        end
        # Assemble full sub matrix
        zerosMatrix = zeros(2,dim)
        return [zerosMatrix; Diagonal(d); zerosMatrix]
    end

    #=
    Jacobi matrices for the tangent space vector basis
    =#
    global function grad_Jx(N)
        l,u = 1,1          # block bandwidths
        λ,μ = 4,4          # sub-block bandwidths: the bandwidths of each block
        cols = rows = 2:4:2(2N+1)  # block sizes
        J = BandedBlockBandedMatrix(0.0im*I, (rows,cols), (l,u), (λ,μ))
        J[1:rows[1], 1:cols[1]] = grad_jacobi_Bx(0)
        if N > 0
            for n = 1:N
                row = sum(rows[1:n])
                col = sum(cols[1:n])
                J[row-rows[n]+1:row, col+1:col+cols[n+1]] = grad_jacobi_Ax(n-1)
                J[row+1:row+rows[n+1], col+1:col+cols[n+1]] = grad_jacobi_Bx(n)
                J[row+1:row+rows[n+1], col-cols[n]+1:col] = grad_jacobi_Cx(n)
            end
        end
        return J
    end

    global function grad_Jy(N)
        l,u = 1,1          # block bandwidths
        λ,μ = 4,4          # sub-block bandwidths: the bandwidths of each block
        cols = rows = 2:4:2(2N+1)  # block sizes
        J = BandedBlockBandedMatrix(0.0im*I, (rows,cols), (l,u), (λ,μ))
        J[1:rows[1], 1:cols[1]] = grad_jacobi_By(0)
        if N > 0
            for n = 1:N
                row = sum(rows[1:n])
                col = sum(cols[1:n])
                J[row-rows[n]+1:row, col+1:col+cols[n+1]] = grad_jacobi_Ay(n-1)
                J[row+1:row+rows[n+1], col+1:col+cols[n+1]] = grad_jacobi_By(n)
                J[row+1:row+rows[n+1], col-cols[n]+1:col] = grad_jacobi_Cy(n)
            end
        end
        return J
    end

    global function grad_Jz(N)
        l,u = 1,1          # block bandwidths
        λ,μ = 2,2          # sub-block bandwidths: the bandwidths of each block
        cols = rows = 2:4:2(2N+1)  # block sizes
        J = BandedBlockBandedMatrix(0.0im*I, (rows,cols), (l,u), (λ,μ))
        J[1:rows[1], 1:cols[1]] = grad_jacobi_Bz(0)
        if N > 0
            for n = 1:N
                row = sum(rows[1:n])
                col = sum(cols[1:n])
                J[row-rows[n]+1:row, col+1:col+cols[n+1]] = grad_jacobi_Az(n-1)
                J[row+1:row+rows[n+1], col+1:col+cols[n+1]] = grad_jacobi_Bz(n)
                J[row+1:row+rows[n+1], col-cols[n]+1:col] = grad_jacobi_Cz(n)
            end
        end
        return J
    end


    #=
    Entries (coefficients) in the Sub-Matrices used in the gradient matrix
    operator functions
    =#

    function grad_coeff_A(l, m, k)
        if l == 0
            return 0
        else
            return betaVal(l, 1) * clebsch_gordan_coeff(l, m, l+1, -1) * spin_func(-1)[k]
        end
    end

    function grad_coeff_B(l, m, k)
        return betaVal(l, -1) * clebsch_gordan_coeff(l, m, l-1, -1) * spin_func(-1)[k]
    end

    function grad_coeff_D(l, m, k)
        return betaVal(l, 1) * clebsch_gordan_coeff(l, m, l+1, 1) * spin_func(1)[k]
    end

    function grad_coeff_E(l, m, k)
        return betaVal(l, -1) * clebsch_gordan_coeff(l, m, l-1, 1) * spin_func(1)[k]
    end

    function grad_coeff_F(l, m, k)
        return betaVal(l, 1) * clebsch_gordan_coeff(l, m, l+1, 0) * spin_func(0)[k]
    end

    function grad_coeff_G(l, m, k)
        return betaVal(l, -1) * clebsch_gordan_coeff(l, m, l-1, 0) * spin_func(0)[k]
    end

    function grad_perp_coeff_A(l, m, k)
        return betaVal(l, 0) * clebsch_gordan_coeff(l, m, l, -1) * spin_func(-1)[k]
    end

    function grad_perp_coeff_D(l, m, k)
        return betaVal(l, 0) * clebsch_gordan_coeff(l, m, l, 1) * spin_func(1)[k]
    end

    function grad_perp_coeff_F(l, m, k)
        if l == 0
            return 0.0
        end
        return betaVal(l, 0) * clebsch_gordan_coeff(l, m, l, 0) * spin_func(0)[k]
    end


    #=
    Sub-Matrices for use in the gradient matrix operator functions
    =#

    function grad_matrix_Ak(n, k)
        dim = 2n+1
        zerosVec = zeros(Complex128, dim)
        if k == 3
            d = copy(zerosVec)
            for j = -n:n
                d[j+n+1] = grad_coeff_F(n, j, k)
            end
            return [zeros(dim, 1) Diagonal(d) zeros(dim, 1)]
        else
            leftdiag = copy(zerosVec)
            rightdiag = copy(zerosVec)
            for j = -n:n
                leftdiag[j+n+1] = grad_coeff_D(n, j, k)
                rightdiag[j+n+1] = grad_coeff_A(n, j, k)
            end
            left = [Diagonal(leftdiag) zeros(dim, 2)]
            right = [zeros(dim, 2) Diagonal(rightdiag)]
            return left + right
        end
    end

    function grad_matrix_Ck(n, k)
        zerosVec = zeros(Complex128, 2n-1)
        if k == 3
            d = copy(zerosVec)
            for j = -n:n-2
                d[j+n+1] = grad_coeff_G(n, j+1, k)
            end
            return [zerosVec'; Diagonal(d); zerosVec']
        else
            upperdiag = copy(zerosVec)
            lowerdiag = copy(zerosVec)
            for j = -n:n-2
                upperdiag[j+n+1] = grad_coeff_B(n, j, k)
                lowerdiag[j+n+1] = grad_coeff_E(n, j+2, k)
            end
            upper = Diagonal(upperdiag)
            lower = Diagonal(lowerdiag)
            return [upper; zerosVec'; zerosVec'] + [zerosVec'; zerosVec'; lower]
        end
    end

    function grad_perp_matrix_Bk(n, k)
        if n == 0
            return grad_perp_coeff_F(0, 0, k)
        end
        upperdiag = zeros(Complex128, 2n)
        lowerdiag = copy(upperdiag)
        d = zeros(Complex128, 2n+1)
        for j = -n:n-1
            upperdiag[j+n+1] = grad_perp_coeff_A(n, j, k)
            d[j+n+1] = grad_perp_coeff_F(n, j, k)
            lowerdiag[j+n+1] = grad_perp_coeff_D(n, j+1, k)
        end
        d[end] = grad_perp_coeff_F(n, n, k)
        return Tridiagonal(lowerdiag, d, upperdiag)
    end

    #=
    Outputs the matrix corresponding to the operator ∂Y/∂x_k where k=1,2,3
    and Y is a spherical harmonic.
    =#
    global function grad_sh(N, k)
        l,u = 1,1          # block bandwidths
        λ,μ = 2,2          # sub-block bandwidths: the bandwidths of each block
        cols = rows = 1:2:2N+1  # block sizes
        J = BandedBlockBandedMatrix(0.0im*I, (rows,cols), (l,u), (λ,μ))
        if N == 0
            return J
        end
        J[1,2:4] = grad_matrix_Ak(0,k)
        J[2:4,1] = grad_matrix_Ck(1,k)
        if N == 1
            return J
        end
        for n = 2:N
            J[(n-1)^2+1:n^2,n^2+1:(n+1)^2] = grad_matrix_Ak(n-1,k)
            J[n^2+1:(n+1)^2,(n-1)^2+1:n^2] = grad_matrix_Ck(n,k)
        end
        return J
    end

    #=
    Outputs the matrix corresponding to the operator that yields the kth entry
    of the vector ∇^⟂(Y_l^m), k=1,2,3 where Y^l_m is the (l,m) spherical
    harmonic.
    =#
    global function grad_perp_sh(N, k)
        l,u = 0,0          # block bandwidths
        λ,μ = 1,1          # sub-block bandwidths: the bandwidths of each block
        cols = rows = 1:2:2N+1  # block sizes
        J = BandedBlockBandedMatrix(0.0im*I, (rows,cols), (l,u), (λ,μ))
        for n = 0:N
            entries = n^2+1:(n+1)^2
            J[entries,entries] = grad_perp_matrix_Bk(n,k)
        end
        return J
    end

    #=
    Matrix representing the Laplacian operator on the vector of the OPs
    (shperical harmonic polynomials) up to order N
    =#
    global function laplacian_sh(N)
        l,u = 0,0
        λ,μ = 0,0
        rows = cols = 1:2:2N+1
        D = BandedBlockBandedMatrix(I, (rows, cols), (l,u), (λ,μ))
        for l=0:N
            entries = l^2+1:(l+1)^2
            D[entries, entries] *= -l*(l+1)
        end
        return D
    end


    #=
    Ouputs the tangent basis vectors (∇Y, ∇⟂Y) up to order N evaluated at a
    point x,y,z on the unit sphere.Returns the vector [∇P_0,...,∇P_N]
    where ∇P_l = [∇Y_l^-l,∇⟂Y_l^-l,...,∇Y_l^l,∇⟂Y_l^l].
    =#
    global function tangent_basis_eval(N, x, y, z)
        Dx = grad_sh(N+1, 1)
        Dy = grad_sh(N+1, 2)
        Dz = grad_sh(N+1, 3)
        DPerpx = grad_perp_sh(N+1, 1)
        DPerpy = grad_perp_sh(N+1, 2)
        DPerpz = grad_perp_sh(N+1, 3)
        Y = sh_eval(N+1, x, y, z)

        len = 6(N+1)^2
        entries = 1:6:len
        out = PseudoBlockArray(Vector{Complex{Float64}}(len), [3 for i=1:2(N+1)^2])
        out[entries] = (Dx*Y)[1:end-(2N+3)]
        out[entries+1] = (Dy*Y)[1:end-(2N+3)]
        out[entries+2] = (Dz*Y)[1:end-(2N+3)]
        out[entries+3] = (DPerpx*Y)[1:end-(2N+3)]
        out[entries+4] = (DPerpy*Y)[1:end-(2N+3)]
        out[entries+5] = (DPerpz*Y)[1:end-(2N+3)]

        return out
    end


    #=
    Functions to output the matrices used in the Clenshaw Algorithm for
    evaluation of a function given by its coefficients of its expansion in the
    tangent basis.

    Note these could be made faster by not calling each grad_jacobi_Bx() etc.
    function (i.e. repeating for loops), and possibly by storing the matrix
    clenshaw_matrix_B(N) and cutting it down to return clenshaw_matrix_B(n) etc.
    =#
    global function clenshaw_matrix_B(n)
        u, uperp, l, lperp = get_Bx_submatrices(n)
        return [sparse(u + uperp + l + lperp); sparse(im*(-u - uperp + l + lperp)); grad_jacobi_Bz(n)]
    end

    global function clenshaw_matrix_G(n, x, y, z)
        iden = speye(2(2n+1))
        return [x*iden; y*iden; z*iden]
    end

    global function clenshaw_matrix_BminusG(n, x, y, z)
        iden = speye(2(2n+1))
        u, uperp, l, lperp = get_Bx_submatrices(n)
        return [sparse(u + uperp + l + lperp) - x*iden; sparse(im*(-u - uperp + l + lperp)) - y*iden; grad_jacobi_Bz(n) - z*iden]
    end

    global function clenshaw_matrix_C(n)
        uppermat, lowermat = get_Cx_submatrices(n)
        return [uppermat+lowermat; im*(-uppermat+lowermat); grad_jacobi_Cz(n)]
    end

    global function clenshaw_matrix_DT(n)
        if n == 0
            return 0.0
        end
        dvec = zeros(2(2n+1))
        avec = zeros(4)
        index = 1
        index2 = 1
        a = d = im
        for k=-n:n-2
            d = coeff_d(n,k)
            view(dvec, index:index+1) .= 0.5/d, 0.5/(d')
            index += 2
        end
        for k=n-1:n
            a = coeff_a(n,k)
            d = coeff_d(n,k)
            view(avec, index2:index2+1) .= 0.5/a, 0.5/(a')
            view(dvec, index:index+1) .= 0.5/d, 0.5/(d')
            index += 2
            index2 += 2
        end
        zerosMatrix = zeros(4,2(2n-1))
        amat = Diagonal(avec)
        dmat = Diagonal(dvec)
        return [dmat -im*dmat zeros(2(2n+1),2(2n+1)); zerosMatrix amat zerosMatrix im*amat zeros(4,2(2n+1))]
    end


    #=
    Function to obtain for storage the coefficient matrices used in the Clenshaw
    Algorithm when using N iterations. Once we have these, we can call the
    tangent_func_eval() method passing in the arrays of matrices to access each
    time, rather than calculating them each time we evaluate.
    =#
    global function get_clenshaw_matrices(N)
        DT = Array{Matrix{Complex128}}(N-1)
        a = copy(DT)
        b = copy(DT)
        for n = N-1:-1:1
            DT[n] = clenshaw_matrix_DT(n)
            a[n] = DT[n] * clenshaw_matrix_B(n)
            b[n] = DT[n] * clenshaw_matrix_C(n)
        end
        return DT, a, b
    end

    #=
    Function(s, overloaded) to obtain the evaluation of a function f(x,y,z)
    where f is input as the coefficients of its expansion in thetangent space
    basis for the sphere, i.e.
        f(x, y, z) = sum(vecdot(f_n, ∇P_n))
    where the {∇P_n} are the basis vectors for the tangent space (gradient and
    perpendicular gradient of the spherical harmonics)

    Uses the Clenshaw Algorithm.
    =#
    global function tangent_func_eval(f, x, y, z, DT, a, b)

        # Check that x and y are on the unit circle
        delta = 0.001
        @assert (x^2 + y^2 + z^2 < 1 + delta &&  x^2 + y^2 + z^2 > 1 - delta) "the point (x, y, z) must be on unit sphere"

        M = length(f)
        N = round(Int, sqrt(M/2) - 1)
        @assert (M > 0 && sqrt(M/2) - 1 == N) "invalid length of f"

        # Complete the reverse recurrance to gain gamma_1, gamma_2
        # Note that gamma_(N+1) = 0, gamma_(N+2) = 0
        if N == 0
            return zeros(Complex128,3)
        elseif N == 1
            gamma_nplus1 = view(f, 2N^2+1:2(N+1)^2).'
        else
            gamma_nplus2 = view(f, 2N^2+1:2(N+1)^2).'
            gamma_nplus1 = (view(f, 2(N-1)^2+1:2N^2).'
                            - gamma_nplus2 * (a[N-1] - DT[N-1]*clenshaw_matrix_G(N-1, x, y, z)))
            for n = N-2:-1:1
                gamma_n = (view(f, 2n^2+1:2(n+1)^2).'
                            - gamma_nplus1 * (a[n] - DT[n]*clenshaw_matrix_G(n, x, y, z))
                            - gamma_nplus2 * b[n+1])
                gamma_nplus2 = copy(gamma_nplus1)
                gamma_nplus1 = copy(gamma_n)
            end
        end

        # Calculate the evaluation of f using gamma_1, gamma_2
        # f(x,y,z) = f_0^T * ∇P_0 + gamma_1 * ∇P_1 - gamma_2 * (DT_1*C_1) * ∇P_0
        # Note ∇P0 = 0
        ∇P1 = tangent_basis_eval(1, x, y, z)
        feval = zeros(Complex128,3)
        Pblock = 3
        for m = -1:1
            feval += gamma_nplus1[Pblock-2] * view(∇P1, Block(Pblock))
            feval += gamma_nplus1[Pblock-1] * view(∇P1, Block(Pblock+1))
            Pblock += 2
        end
        return feval

    end

    global function tangent_func_eval(f, x, y, z)

        # Check that x and y are on the unit circle
        delta = 0.001
        @assert (x^2 + y^2 + z^2 < 1 + delta &&  x^2 + y^2 + z^2 > 1 - delta) "the point (x, y, z) must be on unit sphere"

        M = length(f)
        N = round(Int, sqrt(M/2) - 1)
        @assert (M > 0 && sqrt(M/2) - 1 == N) "invalid length of f"

        # Complete the reverse recurrance to gain gamma_1, gamma_2
        # Note that gamma_(N+1) = 0, gamma_(N+2) = 0
        if N == 0
            return zeros(Complex128,3)
        elseif N == 1
            gamma_nplus1 = view(f, 2N^2+1:2(N+1)^2).'
        else
            gamma_nplus2 = view(f, 2N^2+1:2(N+1)^2).'
            DT = clenshaw_matrix_DT(N-1)
            gamma_nplus1 = view(f, 2(N-1)^2+1:2N^2).' - gamma_nplus2 * DT * clenshaw_matrix_BminusG(N-1, x, y, z)
            for n = N-2:-1:1
                b = DT * clenshaw_matrix_C(n+1)
                DT = clenshaw_matrix_DT(n)
                a = DT * clenshaw_matrix_BminusG(n, x, y, z)
                gamma_n = view(f, 2n^2+1:2(n+1)^2).' - gamma_nplus1 * a - gamma_nplus2 * b
                gamma_nplus2 = copy(gamma_nplus1)
                gamma_nplus1 = copy(gamma_n)
            end
        end

        # Calculate the evaluation of f using gamma_1, gamma_2
        # f(x,y,z) = f_0^T * ∇P_0 + gamma_1 * ∇P_1 - gamma_2 * (DT_1*C_1) * ∇P_0
        # Note ∇P0 = 0
        ∇P1 = tangent_basis_eval(1, x, y, z)
        feval = zeros(Complex128,3)
        Pblock = 3
        for m = -1:1
            feval += gamma_nplus1[Pblock-2] * view(∇P1, Block(Pblock))
            feval += gamma_nplus1[Pblock-1] * view(∇P1, Block(Pblock+1))
            Pblock += 2
        end
        return feval

    end

    #=
    Function to obtain the matrix evaluation of a function f(x,y,z) with inputs
    (J^x_∇,J^y_∇,J^z_∇) where f is input as the coefficients of its expansion
    in the basis of the OPs for the sphere, i.e.
        f(x, y) = sum(vecdot(f_n, P_n))
    where the {P_n} are the OPs on the sphere (spherical harmonics)

    Uses the Clenshaw Algorithm.

    # THIS IS WRONG
    =#
    global function func_eval_grad_jacobi(f)
        # Define the Jacobi operator matrices and pass to evaluation function
        return func_eval_operator(f, grad_Jx(N), grad_Jy(N), grad_Jz(N))
    end

end



#====#
# Testing


tol = 1e-10



N = 10
Dx = grad_sh(N, 1)
Dy = grad_sh(N, 2)
Dz = grad_sh(N, 3)
# I would expect this to just be diagonal (like the Laplacian)
D2 = Dx^2 + Dy^2 + Dz^2
Lap = laplacian_sh(N)
# D2 then matches Lap (ignoring the last 2N+1 rows/cols, since the matrices for
# Dx are not true representations for derivatives at these entries)
B = abs.(D2 - Lap)[1:end-(2N+1), 1:end-(2N+1)]
@test count(i->i>tol, B) == 0

DPerpx = grad_perp_sh(N, 1)
DPerpy = grad_perp_sh(N, 2)
DPerpz = grad_perp_sh(N, 3)
D2Perp = DPerpx^2 + DPerpy^2 + DPerpz^2
# D2 then matches Lap (ignoring the last 2N+1 rows/cols, since the matrices for
# Dx are not true representations for derivatives at these entries)
B = abs.(D2Perp - Lap)[1:end-(2N+1), 1:end-(2N+1)]
@test count(i->i>tol, B) == 0

######

x, y = 0.5, 0.1
z = sqrt(1 - x^2 - y^2)
Y = sh_eval(N, x, y, z)
DxY = Dx*Y
DPerpxY = DPerpx*Y
for l = 1:6
    for m = -l:l
        @test abs(x*DxY[l^2+l+1+m] - (coeff_a(l, m)*DxY[(l+1)^2+l+1+1+m+1]
                                        + coeff_b(l, m)*DxY[(l-1)^2+l-1+1+m+1]
                                        + coeff_d(l, m)*DxY[(l+1)^2+l+1+1+m-1]
                                        + coeff_e(l, m)*DxY[max(1,(l-1)^2+l-1+1+m-1)]
                                        + coeff_h(l, m)*DPerpxY[l^2+l+1+m+1]
                                        + coeff_j(l, m)*DPerpxY[l^2+l+1+m-1])
            ) < tol
        @test abs(y*DxY[l^2+l+1+m] - (- im * coeff_a(l, m)*DxY[(l+1)^2+l+1+1+m+1]
                                        - im * coeff_b(l, m)*DxY[(l-1)^2+l-1+1+m+1]
                                        + im * coeff_d(l, m)*DxY[(l+1)^2+l+1+1+m-1]
                                        + im * coeff_e(l, m)*DxY[max(1,(l-1)^2+l-1+1+m-1)]
                                        - im * coeff_h(l, m)*DPerpxY[l^2+l+1+m+1]
                                        + im * coeff_j(l, m)*DPerpxY[l^2+l+1+m-1])
            ) < tol
        @test abs(z*DxY[l^2+l+1+m] - (coeff_f(l, m)*DxY[(l+1)^2+l+1+1+m]
                                        + coeff_g(l, m)*DxY[max(1, (l-1)^2+l-1+1+m)]
                                        + coeff_k(l, m)*DPerpxY[l^2+l+1+m])
            ) < tol

        @test abs(x*DPerpxY[l^2+l+1+m] - (perp_coeff_a(l, m)*DPerpxY[(l+1)^2+l+1+1+m+1]
                                        + perp_coeff_b(l, m)*DPerpxY[(l-1)^2+l-1+1+m+1]
                                        + perp_coeff_d(l, m)*DPerpxY[(l+1)^2+l+1+1+m-1]
                                        + perp_coeff_e(l, m)*DPerpxY[max(1,(l-1)^2+l-1+1+m-1)]
                                        + perp_coeff_h(l, m)*DxY[l^2+l+1+m+1]
                                        + perp_coeff_j(l, m)*DxY[l^2+l+1+m-1])
            ) < tol
        @test abs(y*DPerpxY[l^2+l+1+m] - (- im * perp_coeff_a(l, m)*DPerpxY[(l+1)^2+l+1+1+m+1]
                                            - im * perp_coeff_b(l, m)*DPerpxY[(l-1)^2+l-1+1+m+1]
                                            + im * perp_coeff_d(l, m)*DPerpxY[(l+1)^2+l+1+1+m-1]
                                            + im * perp_coeff_e(l, m)*DPerpxY[max(1,(l-1)^2+l-1+1+m-1)]
                                            - im * perp_coeff_h(l, m)*DxY[l^2+l+1+m+1]
                                            + im * perp_coeff_j(l, m)*DxY[l^2+l+1+m-1])
            ) < tol
        @test abs(z*DPerpxY[l^2+l+1+m] - (perp_coeff_f(l, m)*DPerpxY[(l+1)^2+l+1+1+m]
                                            + perp_coeff_g(l, m)*DPerpxY[max(1, (l-1)^2+l-1+1+m)]
                                            + perp_coeff_k(l, m)*DxY[l^2+l+1+m])
            ) < tol
    end
end


#####


# a = x*tangent_basis_eval(N,x,y,z)
# b = grad_Jx(N)*tangent_basis_eval(N,x,y,z)
# c = abs.(a[1:6N^2] - b[1:6N^2])
# @test count(i->i>tol, c) == 0
# a = y*tangent_basis_eval(N,x,y,z)
# b = grad_Jy(N)*tangent_basis_eval(N,x,y,z)
# c = abs.(a[1:6N^2] - b[1:6N^2])
# @test count(i->i>tol, c) == 0
# a = z*tangent_basis_eval(N,x,y,z)
# b = grad_Jz(N)*tangent_basis_eval(N,x,y,z)
# c = abs.(a[1:6N^2] - b[1:6N^2])
# @test count(i->i>tol, c) == 0


#####


N = 5
f = 2*ones(2(N+1)^2)
∇P1 = tangent_basis_eval(N,x,y,z)
feval = tangent_func_eval(f,x,y,z)
feval_actual = zeros(3)
for i=1:length(f)
        feval_actual += f[i] * view(∇P1, Block(i))
end
@test feval_actual ≈ feval
