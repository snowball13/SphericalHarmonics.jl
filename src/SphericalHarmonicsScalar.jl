# Script to obtain a point evaluation of the OPs for the unit sphere (spherical harmonics)

let

    #=
    Functions to obtain the coefficients used in the matrices for
    J^x, J^y, J^z
    =#

    # Function outputting the constant for the (l,m) spherical harmonic
    # polynomial
    global function alphaVal(l, m)
        α = 0
        if m == 0
            α = sqrt((2l+1)/(4pi))
        elseif m > 0
            α = sqrt((2l+1)/(4pi) * gamma(big(l+m+1)) * gamma(big(l-m+1))) / (gamma(big(l+1)) * (-2.0)^m)
        else
            m = abs(m)
            α = sqrt((2l+1)/(4pi) * gamma(big(l+m+1)) * gamma(big(l-m+1))) / (gamma(big(l+1)) * (2.0)^m)
        end
        return Float64(α)
    end

    global function AtildeVal(l, m)
        return (l+m+2)*(l+m+1) / (2*(2l+1)*(l+1))
    end

    global function BtildeVal(l, m)
        return -l/(4l+2)
    end

    global function DtildeVal(l, m)
        return - 2*(l-m+2)*(l-m+1) / ((2l+1)*(l+1))
    end

    global function EtildeVal(l, m)
        return 2l/(2l+1)
    end

    global function FtildeVal(l, m)
        return (l-m+1)*(l+m+1) / ((2l+1)*(l+1))
    end

    global function GtildeVal(l, m)
        G = 1.0
        if (abs(m) <= l - 1)
            G = l / (2l + 1)
        else
            G = 0.0
        end
        return G
    end

    # The following coeff functions give the coefficients used in the
    # relations for x*Y^m_l, y*Y^m_l and z*Y^m_l where Y^m_l is the l,m
    # spherical harmonic polynomial. These are then used as the non-zero entries
    # in our system matrices.
    global function coeff_A(l, m)
        # Do we have a non-zero coeff?
        if abs(m+1) > l+1
            return 0.0
        end
        A = 1.0
        if (m >= 0)
            A = AtildeVal(l, m)
        else
            A = DtildeVal(l, abs(m))
        end
        A *= alphaVal(l, m) / (2 * alphaVal(l+1, m+1))
        return A
    end

    global function coeff_B(l, m)
        # Do we have a non-zero coeff?
        if abs(m+1) > l-1
            return 0.0
        end
        B = 1.0
        if (m >= 0)
            if (l - abs(m) - 2 >= 0)
                B = BtildeVal(l, m)
            else
                return 0.0
            end
        else
            B = EtildeVal(l, abs(m))
        end
        B *= alphaVal(l, m) / (2 * alphaVal(l-1, m+1))
        return B
    end

    global function coeff_D(l, m)
        # Do we have a non-zero coeff?
        if abs(m-1) > l+1
            return 0.0
        end
        D = 1.0
        if (m > 0)
            D = DtildeVal(l, m)
        else
            D = AtildeVal(l, abs(m))
        end
        D *= alphaVal(l, m) / (2 * alphaVal(l+1, m-1))
        return D
    end

    global function coeff_E(l, m)
        # Do we have a non-zero coeff?
        if abs(m-1) > l-1
            return 0.0
        end
        E = 1.0
        if (m > 0)
            E = EtildeVal(l, m)
        else
            if (l - abs(m) - 2 >= 0)
                E = BtildeVal(l, abs(m))
            else
                return 0.0
            end
        end
        E *= alphaVal(l, m) / (2 * alphaVal(l-1, m-1))
        return E
    end

    global function coeff_F(l, m)
        # Do we have a non-zero coeff?
        if abs(m) > l+1
            return 0.0
        end
        return FtildeVal(l, abs(m)) * alphaVal(l, m) / alphaVal(l+1, m)
    end

    global function coeff_G(l, m)
        # Do we have a non-zero coeff?
        if abs(m) > l-1
            return 0.0
        end
        return GtildeVal(l, abs(m)) * alphaVal(l, m) / alphaVal(l-1, m)
    end


    #===#


    #=
    Functions to obtain the matrices used in the 3-term relation for evaluating
    the OPs on the sphere
    =#

    function systemMatrix_A(n)
        A = 0
        if n == 0
            d_00 = coeff_D(0, 0)
            a_00 = coeff_A(0, 0)
            f_00 = coeff_F(0, 0)
            A = [d_00 0 a_00; im*d_00 0 -im*a_00; 0 f_00 0]
        else
            # We proceed by creating diagonal matrices using the coefficients of
            # the 3-term relations for the spherical harmonic OPs, and then combining
            zerosVec = zeros(2*n + 1)
            leftdiag = copy(zerosVec)
            rightdiag = copy(zerosVec)
            lowerdiag = copy(zerosVec)
            for k = -n:n
                leftdiag[k+n+1] = coeff_D(n, k)
                rightdiag[k+n+1] = coeff_A(n, k)
                lowerdiag[k+n+1] = coeff_F(n, k)
            end
            left = [Diagonal(leftdiag) zeros(2*n+1, 2)]
            right = [zeros(2*n+1, 2) Diagonal(rightdiag)]
            lower = [zeros(2*n+1, 1) Diagonal(lowerdiag) zeros(2*n+1, 1)]
            A = [left + right; -im*(-left + right); lower]
        end
        return A
    end

    function systemMatrix_B(n)
        return zeros(3*(2*n + 1), 2*n + 1)
    end

    function systemMatrix_C(n)
        if n == 0
            return zeros(3, 1)
        elseif n == 1
            b_11 = coeff_B(1, -1)
            e_11 = coeff_E(1, 1)
            g_10 = coeff_G(1, 0)
            return [b_11; 0; e_11; -im*b_11; 0; im*e_11; 0; g_10; 0]
        end
        # We proceed by creating diagonal matrices using the coefficients of
        # the 3-term relations for the spherical harmonic OPs, and then combining
        zerosVec = zeros(2*n - 1)
        upperdiag = copy(zerosVec)
        lowerdiag = copy(zerosVec)
        diag_z = copy(zerosVec)
        for k = -n:n-2
            upperdiag[k+n+1] = coeff_B(n, k)
            lowerdiag[k+n+1] = coeff_E(n, k+2)
            diag_z[k+n+1] = coeff_G(n, k+1)
        end
        upper = Diagonal(upperdiag)
        lower = Diagonal(lowerdiag)
        C_x = [upper; zerosVec'; zerosVec'] + [zerosVec'; zerosVec'; lower]
        C_y = [upper; zerosVec'; zerosVec'] - [zerosVec'; zerosVec'; lower]
        C_y *= -im
        C_z = [zerosVec'; Diagonal(diag_z); zerosVec']
        return [C_x; C_y; C_z]
    end

    function systemMatrix_G(n, x, y, z)
        Iden = speye(2*n + 1)
        return [x*Iden; y*Iden; z*Iden]
    end

    function systemMatrix_DT(n)
        # Note DT_n is a right inverse matrix of A_n
        DT = 0
        if n == 0
            d_00 = coeff_D(0, 0)
            a_00 = coeff_A(0, 0)
            f_00 = coeff_F(0, 0)
            DT = [1./(2*d_00) -im/(2*d_00) 0; 0 0 1./f_00; 1./(2*a_00) im/(2*a_00) 0]
        else
            # We proceed by creating diagonal matrices using the coefficients of
            # the 3-term relations for the spherical harmonic OPs, and then combining
            upperdiag = zeros(2*n + 1)
            for k = -n:n
                upperdiag[k+n+1] = 1./(2 * coeff_D(n, k))
            end
            upper = [Diagonal(upperdiag) Diagonal(-im*upperdiag) zeros(2*n+1, 2*n+1)]
            lower = im*zeros(2, 3*(2*n+1))
            lower[1, 2*n] = 1./(2 * coeff_A(n, n-1))
            lower[2, 2*n+1] = 1./(2 * coeff_A(n, n))
            lower[1, 2*(2*n+1)-1] = im*lower[1, 2*n]
            lower[2, 2*(2*n+1)] = im*lower[2, 2*n+1]
            DT = [upper; lower]
        end
        return DT
    end


    #===#


    #=
    Functions to obtain the matrices used in Clenshaw Algorithm when evaluating
    a function at the Jacobi operator matrices Jx,Jy,Jz. These are required to
    not be sparse due to Julia constraints/errors thrown when using multiplying
    sparse matrices with type Matrix{Matrix{Float64}}.
    =#

    function systemMatrix_C_operator(n)
        if n == 0
            return zeros(3, 1)
        elseif n == 1
            b_11 = coeff_B(1, -1)
            e_11 = coeff_E(1, 1)
            g_10 = coeff_G(1, 0)
            return [b_11; 0; e_11; -im*b_11; 0; im*e_11; 0; g_10; 0]
        end
        # We proceed by creating diagonal matrices using the coefficients of
        # the 3-term relations for the spherical harmonic OPs, and then combining
        zerosVec = zeros(2*n - 1)
        upperdiag = copy(zerosVec)
        lowerdiag = copy(zerosVec)
        diag_z = copy(zerosVec)
        for k = -n:n-2
            upperdiag[k+n+1] = coeff_B(n, k)
            lowerdiag[k+n+1] = coeff_E(n, k+2)
            diag_z[k+n+1] = coeff_G(n, k+1)
        end
        upper = diagm(upperdiag)
        lower = diagm(lowerdiag)
        C_x = [upper; zerosVec'; zerosVec'] + [zerosVec'; zerosVec'; lower]
        C_y = [upper; zerosVec'; zerosVec'] - [zerosVec'; zerosVec'; lower]
        C_y *= -im
        C_z = [zerosVec'; diagm(diag_z); zerosVec']
        return [C_x; C_y; C_z]
    end

    function systemMatrix_DT_operator(n)
        # Note DT_n is a right inverse matrix of A_n
        DT = 0
        if n == 0
            d_00 = coeff_D(0, 0)
            a_00 = coeff_A(0, 0)
            f_00 = coeff_F(0, 0)
            DT = [1./(2*d_00) -im/(2*d_00) 0; 0 0 1./f_00; 1./(2*a_00) im/(2*a_00) 0]
        else
            # We proceed by creating diagonal matrices using the coefficients of
            # the 3-term relations for the spherical harmonic OPs, and then combining
            upperdiag = zeros(2*n + 1)
            for k = -n:n
                upperdiag[k+n+1] = 1./(2 * coeff_D(n, k))
            end
            upper = [diagm(upperdiag) diagm(-im*upperdiag) zeros(2*n+1, 2*n+1)]
            lower = im*zeros(2, 3*(2*n+1))
            lower[1, 2*n] = 1./(2 * coeff_A(n, n-1))
            lower[2, 2*n+1] = 1./(2 * coeff_A(n, n))
            lower[1, 2*(2*n+1)-1] = im*lower[1, 2*n]
            lower[2, 2*(2*n+1)] = im*lower[2, 2*n+1]
            DT = [upper; lower]
        end
        return DT
    end

    function systemMatrix_G_operator(n, J_x, J_y, J_z)
        G = Matrix{Matrix{Complex{Float64}}}(3*(2n+1),2n+1)
        for i=1:2n+1
            for j=1:2n+1
                if i == j
                    G[i,j] = J_x
                    G[i+2n+1,j] = J_y
                    G[i+4n+2,j] = J_z
                else
                    G[i,j] = zeros(J_x)
                    G[i+2n+1,j] = zeros(J_y)
                    G[i+4n+2,j] = zeros(J_z)
                end
            end
        end
        return G
    end


    #====#


    #=
    Functions to obtain the matrices corresponding to multiplication of the OPs
    by x, y and z respectively (i.e. the Jacobi operators J^x, J^y and J^z)
    =#

    function systemMatrix_Ax(n)
        zerosVec = zeros(2*n + 1)
        leftdiag = copy(zerosVec)
        rightdiag = copy(zerosVec)
        for k = -n:n
            leftdiag[k+n+1] = coeff_D(n, k)
            rightdiag[k+n+1] = coeff_A(n, k)
        end
        left = [Diagonal(leftdiag) zeros(2*n+1, 2)]
        right = [zeros(2*n+1, 2) Diagonal(rightdiag)]
        return left + right
    end

    function systemMatrix_Ay(n)
        zerosVec = zeros(2*n + 1)
        leftdiag = copy(zerosVec)
        rightdiag = copy(zerosVec)
        for k = -n:n
            leftdiag[k+n+1] = coeff_D(n, k)
            rightdiag[k+n+1] = coeff_A(n, k)
        end
        left = [Diagonal(leftdiag) zeros(2*n+1, 2)]
        right = [zeros(2*n+1, 2) Diagonal(rightdiag)]
        return -im*(-left + right)
    end

    function systemMatrix_Az(n)
        zerosVec = zeros(2*n + 1)
        d = copy(zerosVec)
        for k = -n:n
            d[k+n+1] = coeff_F(n, k)
        end
        return [zeros(2*n+1, 1) Diagonal(d) zeros(2*n+1, 1)]
    end

    function systemMatrix_Bx(n)
        return zeros(2*n+1, 2*n+1)
    end

    function systemMatrix_By(n)
        return zeros(2*n+1, 2*n+1)
    end

    function systemMatrix_Bz(n)
        return zeros(2*n+1, 2*n+1)
    end

    function systemMatrix_Cx(n)
        zerosVec = zeros(2*n - 1)
        upperdiag = copy(zerosVec)
        lowerdiag = copy(zerosVec)
        for k = -n:n-2
            upperdiag[k+n+1] = coeff_B(n, k)
            lowerdiag[k+n+1] = coeff_E(n, k+2)
        end
        upper = Diagonal(upperdiag)
        lower = Diagonal(lowerdiag)
        return [upper; zerosVec'; zerosVec'] + [zerosVec'; zerosVec'; lower]
    end

    function systemMatrix_Cy(n)
        zerosVec = zeros(2*n - 1)
        upperdiag = copy(zerosVec)
        lowerdiag = copy(zerosVec)
        for k = -n:n-2
            upperdiag[k+n+1] = coeff_B(n, k)
            lowerdiag[k+n+1] = coeff_E(n, k+2)
        end
        upper = Diagonal(upperdiag)
        lower = Diagonal(lowerdiag)
        return - im * ([upper; zerosVec'; zerosVec'] - [zerosVec'; zerosVec'; lower])
    end

    function systemMatrix_Cz(n)
        zerosVec = zeros(2*n - 1)
        d = copy(zerosVec)
        for k = -n:n-2
            d[k+n+1] = coeff_G(n, k+1)
        end
        return [zerosVec'; Diagonal(d); zerosVec']
    end

    global function Jx(N)
        l,u = 1,1          # block bandwidths
        λ,μ = 2,2         # sub-block bandwidths: the bandwidths of each block
        cols = rows = 1:2:2N+1  # block sizes
        J = BandedBlockBandedMatrix(0.0im*I, (rows,cols), (l,u), (λ,μ))
        if N == 0
            return J
        end
        J[1,2:4] = systemMatrix_Ax(0)
        J[2:4,1] = systemMatrix_Cx(1)
        if N == 1
            return J
        end
        for n = 2:N
            J[(n-1)^2+1:n^2,n^2+1:(n+1)^2] = systemMatrix_Ax(n-1)
            J[n^2+1:(n+1)^2,(n-1)^2+1:n^2] = systemMatrix_Cx(n)
        end
        return J
    end

    global function Jy(N)
        l,u = 1,1          # block bandwidths
        λ,μ = 2,2         # sub-block bandwidths: the bandwidths of each block
        cols = rows = 1:2:2N+1  # block sizes
        J = BandedBlockBandedMatrix(0.0im*I, (rows,cols), (l,u), (λ,μ))
        if N == 0
            return J
        end
        J[1,2:4] = systemMatrix_Ay(0)
        J[2:4,1] = systemMatrix_Cy(1)
        if N == 1
            return J
        end
        for n = 2:N
            J[(n-1)^2+1:n^2,n^2+1:(n+1)^2] = systemMatrix_Ay(n-1)
            J[n^2+1:(n+1)^2,(n-1)^2+1:n^2] = systemMatrix_Cy(n)
        end
        return J
    end

    global function Jz(N)
        l,u = 1,1          # block bandwidths
        λ,μ = 2,2         # sub-block bandwidths: the bandwidths of each block
        cols = rows = 1:2:2N+1  # block sizes
        J = BandedBlockBandedMatrix(0.0im*I, (rows,cols), (l,u), (λ,μ))
        if N == 0
            return J
        end
        J[1,2:4] = systemMatrix_Az(0)
        J[2:4,1] = systemMatrix_Cz(1)
        if N == 1
            return J
        end
        for n = 2:N
            J[(n-1)^2+1:n^2,n^2+1:(n+1)^2] = systemMatrix_Az(n-1)
            J[n^2+1:(n+1)^2,(n-1)^2+1:n^2] = systemMatrix_Cz(n)
        end
        return J
    end


    #====#


    #=
    Functions to obtain the point evaluation of the Nth set of OPs (order N) at
    the point on the unit sphere (x, y, z)
    =#
    global function sh_eval(N, x, y, z)

        # Check that x and y are on the unit circle
        delta = 0.001
        @assert (x^2 + y^2 + z^2 < 1 + delta &&  x^2 + y^2 + z^2 > 1 - delta) "the point (x, y) must be on unit circle"

        # Check that N is a non-negative integer
        @assert N >= 0 "the argument N should be a non-negative integer"

        # We initialise P_(-1) = 0, P_0 = 1, and an empty vector for P_1
        P_nminus1 = 0
        P_n = alphaVal(0, 0)
        P_nplus1 = 0
        P = zeros((N+1)^2)+0im
        P[1] = P_n

        for n = 0:N-1
            # Define the matrices in the 3-term relation
            B_n = systemMatrix_B(n)
            C_n = systemMatrix_C(n)
            G_n = systemMatrix_G(n, x, y, z)
            DT_n = systemMatrix_DT(n)

            # Calculate the next set of OPs
            P_nplus1 = - DT_n * (B_n - G_n) * P_n - DT_n * C_n * P_nminus1

            # Re-label for the next step
            P_nminus1 = copy(P_n)
            P_n = copy(P_nplus1)
            P[(n+1)^2+1:(n+2)^2] = P_n
        end

        return P

    end

    global function sh_eval(l, m, x, y, z)
        # Only return the l,m spherical harmonic OP evaluation.
        P = sh_eval(l,x,y,z)
        return P[l^2+l+1+m]
    end


    #====#


    #=
    Function to obtain a point evaluation of a function f(x,y,z) where f is input as
    the coefficients of its expansion in the basis of the OPs for the sphere, i.e.
        f(x, y) = sum(vecdot(f_n, P_n))
    where the {P_n} are the OPs on the sphere (spherical harmonics)

    Uses the Clenshaw Algorithm.
    =#
    global function func_eval(f, x, y, z)

        # Check that x and y are on the unit circle
        delta = 0.001
        @assert (x^2 + y^2 + z^2 < 1 + delta &&  x^2 + y^2 + z^2 > 1 - delta) "the point (x, y, z) must be on unit sphere"

        M = length(f)
        N = round(Int, sqrt(M) - 1)
        @assert (M > 0 && sqrt(M) - 1 == N) "invalid length of f"

        # Complete the reverse recurrance to gain gamma_1, gamma_2
        # Note that gamma_(N+1) = 0, gamma_(N+2) = 0
        gamma_nplus2 = zeros((N+3)^2-(N+2)^2)
        gamma_nplus1 = zeros((N+2)^2-(N+1)^2)
        gamma_n = 0.0
        for n = N:-1:1
            a = - (systemMatrix_DT(n) * (systemMatrix_B(n) - systemMatrix_G(n, x, y, z))).'
            b = - (systemMatrix_DT(n+1) * systemMatrix_C(n+1)).'
            gamma_n = view(f, n^2+1:(n+1)^2) + a * gamma_nplus1 + b * gamma_nplus2
            gamma_nplus2 = copy(gamma_nplus1)
            gamma_nplus1 = copy(gamma_n)
        end

        # Calculate the evaluation of f using gamma_1, gamma_2
        # f(x,y,z) = P_0*f_0 + gamma_1^T * P_1 - (DT_1*C_1)^T * gamma_2
        b = - (systemMatrix_DT(1) * systemMatrix_C(1)).'
        P_1 = sh_eval(1, x, y, z)
        P_0 = sh_eval(0, 0, x, y, z)
        #return P_0 * f[1] + vecdot(gamma_nplus1, P_1) + P_0 * b * gamma_nplus2
        return P_0 * f[1] + (P_1[2:end].' * gamma_nplus1)[1] + P_0 * b * gamma_nplus2

    end

    #=
    Function to obtain the matrix evaluation of a function f(x,y,z) with inputs
    (Jx,Jy,Jz) where f is input as the coefficients of its expansion in the
    basis of the OPs for the sphere, i.e.
        f(x, y) = sum(vecdot(f_n, P_n))
    where the {P_n} are the OPs on the sphere (spherical harmonics)

    Uses the Clenshaw Algorithm.

    Here, J_x etc can be either the jacobi matrices acting on the SHs or the
    jacobi matrices acting on the grad (tangent) basis.
    =#
    global function func_eval_operator(f, J_x, J_y, J_z)

        M = length(f)
        N = round(Int, sqrt(M) - 1)
        @assert (M > 0 && sqrt(M) - 1 == N) "invalid length of f"

        # Define a zeros vector to store the gammas.
        # Note that we add in gamma_(N+1) = 0, gamma_(N+2) = 0
        gamma_nplus2 = Vector{Matrix{Float64}}(2N+5)
        gamma_nplus1 = Vector{Matrix{Float64}}(2N+3)
        gamma_n = Vector{Matrix{Float64}}(2N+1)
        for k in eachindex(gamma_nplus1)
            gamma_nplus2[k] = zeros(J_x)
            gamma_nplus1[k] = zeros(J_x)
        end
        gamma_nplus2[end-1] = zeros(J_x)
        gamma_nplus2[end] = zeros(J_x)

        # Complete the reverse recurrance to gain gamma_1, gamma_2
        for n = N:-1:1
            a = systemMatrix_B(n) - systemMatrix_G_operator(n,J_x,J_y,J_z)
            a = - systemMatrix_DT_operator(n) * a
            b = - systemMatrix_DT_operator(n+1) * systemMatrix_C_operator(n+1)
            gamma_n = view(f, n^2+1:(n+1)^2).*I + a.' * gamma_nplus1 + b.' * gamma_nplus2
            gamma_nplus2 = copy(gamma_nplus1)
            gamma_nplus1 = copy(gamma_n)
        end

        # Calculate the evaluation of f using gamma_1, gamma_2
        b = - systemMatrix_DT_operator(1) * systemMatrix_C_operator(1)
        P_1 = [alphaVal(1,-1)*(J_x-im*J_y), alphaVal(1,0)*J_z, alphaVal(1,1)*(J_x+im*J_y)]
        P_0 = alphaVal(0,0)
        return P_0 * f[1].*I + P_1.' * gamma_nplus1 + P_0 * b.' * gamma_nplus2

    end

    #=
    Function to obtain the matrix evaluation of a function f(x,y,z) with inputs
    (Jx,Jy,Jz) where f is input as the coefficients of its expansion in the
    basis of the OPs for the sphere, i.e.
        f(x, y) = sum(vecdot(f_n, P_n))
    where the {P_n} are the OPs on the sphere (spherical harmonics)

    Uses the Clenshaw Algorithm.
    =#
    global function func_eval_jacobi(f)
        # Define the Jacobi operator matrices and pass to evaluation function
        return func_eval_operator(f, Jx(N), Jy(N), Jz(N))
    end

end


#-----
# Testing

x = 0.1
y = 0.8
z = sqrt(1 - x^2 - y^2)
l,m = 3,-1
p = sh_eval(l, m, x, y, z)
p_actual = alphaVal(l,m) * (x - im*y) * ((z-1)^2 * 15/4 + (z-1) * 15/2 + 3)
# p_actual = alphaVal(2,2) * (x + im*y)^2
@test p ≈ p_actual

N = 10
f = 1:(N+1)^2
fxyz = func_eval(f, x, y, z)
p = sh_eval(N, x, y, z)
fxyz_actual = 0.0
for k = 0:N
    fxyz_actual += vecdot(view(f, k^2+1:(k+1)^2), view(p, k^2+1:(k+1)^2))
end
fxyz_actual
@test fxyz ≈ fxyz_actual

N = 5
f = 1:(N+1)^2
fxyz = func_eval_jacobi(f)

a = y*sh_eval(N,x,y,z)
b = Jy(N)*sh_eval(N,x,y,z)
(a - b)[1:N^2]
@test a[1:N^2]≈b[1:N^2]
