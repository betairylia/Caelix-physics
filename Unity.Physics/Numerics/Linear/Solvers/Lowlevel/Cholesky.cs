using Unity.Burst;
using Unity.Collections;
using Unity.Mathematics;

namespace Unity.Numerics.Linear.Dense.Primitives
{
    [BurstCompile]
    [GenerateTestsForBurstCompatibility]
    internal unsafe struct Cholesky
    {
        /// <summary>
        /// Compute Cholesky factorization for a symmetric, positive definite matrix A:
        ///   A = L*L^T
        /// where L is lower triangular.
        /// </summary>
        /// <param name="A">The matrix to factor.</param>
        /// <param name="L">The produced L factor.</param>
        /// <param name="singularRow">-1 if algorithm succeeded. Otherwise, contains row index at which matrix was found to be singular. </param>
        public static void Factor(in Matrix A, in Matrix L, out int singularRow)
        {
            var n = A.MinDimension;
            if (L.MinDimension < n)
            {
                UnityEngine.Debug.LogError("Matrix L has invalid dimensions.");
            }

            singularRow = -1;

            L.Clear();
            L.SetDiagonal(0, 1, false, false, true);

            for(int i = 0; i < n; ++i)
            {
                for(int j = 0; j < i; ++j)
                {
                    L[i, j] = (A[i, j] - L.Rows[i].Subvector(0, i).Dot(L.Rows[j].Subvector(0, i))) / L[j, j];
                }
                var x = A[i, i] - L.Rows[i].Subvector(0, i).Dot(L.Rows[i].Subvector(0, i));
                if (x <= 0)
                {
                    singularRow = i;
                    return;
                }
                L[i, i] = math.sqrt(x);
            }
        }

        /// <summary>
        /// Compute in-place Cholesky factorization for a symmetric, positive definite matrix A:
        ///   A = L*L^T
        /// where L is lower triangular.
        /// </summary>
        /// <param name="A">The matrix to factor. Lower triangular part will be overwritten with L factor.</param>
        /// <param name="singularRow">-1 if algorithm succeeded. Otherwise, contains row index at which matrix was found to be singular. </param>
        public static void Factor(in Matrix A, out int singularRow)
        {
            singularRow = -1;

            var n = A.MinDimension;
            for(int k = 0; k < n; ++k)
            {
                int rs = n - (k+1); // remaining size

                using var A21 = A.Submatrix(k+1, k, rs, 1);
                using var A10 = A.Submatrix(k, 0, 1, k);
                using var A20 = A.Submatrix(k+1, 0, rs, k);

                var x = A[k,k];
                if (k > 0)
                {
                    x -= A10.Rows[0].Dot(A10.Rows[0]); // squared norm: ||A10||^2
                }

                if (x <= 0)
                {
                    singularRow = k;
                    return;
                }

                x = math.sqrt(x);
                A[k, k] = x;
                if (k > 0 && rs > 0)
                {
                    A21.ScaleAndAddProduct(Op.None, Op.Transpose, -1, A20, A10, 1); // A21 -= A20 * A10^tr;
                }

                if (rs > 0)
                {
                    A21.ScaleAndAddProduct(Op.None, Op.None, 0, A21, A21, 1/x); // A21 /= x;
                }
            }
        }

        // Downdate a Cholesky factor L_FF corresponding to the Cholesky decomposition of
        // the principal submatrix A_FF with index set F \in {1...n} of a full system matrix A based on
        // the full index set, by adding the given index i to F, producing the new Cholesky factor
        // L_{F'F'}, with F' = F - {i}, for A_{F'F'}, that is
        //
        //              L_{F'F'} * L_{F'F'}^T = A_{F'F'}.
        //
        // The operation performed here is equivalent to the rank-1 update A_{F'F'} = A_{FF} - x*x^T with
        // x = L[i, :] being the i'th column of L.
        //
        // Assumes that L_FF has dimensions corresponding to a factorization for the full system A which includes all indices.
        public static void Downdate(in Matrix L_FF, int index)
        {
            int n = L_FF.MinDimension;
            int n_1 = n - 1;
            int rows = n_1 - index;
            var w = L_FF.Cols[index].Subvector(n-rows, rows);
            var beta = 1f;
            for(int j = 0; j < rows; ++j)
            {
                int k = index + j + 1;
                var x_j = w[j];
                var x_j_sq = x_j*x_j;
                var y = L_FF[k, k];
                var ysq = y*y;
                var z = ysq + x_j_sq / beta;
                var gamma = ysq*beta + x_j_sq;
                var sqrt_z = math.sqrt(z);
                L_FF[k, k] = sqrt_z;
                beta += x_j_sq / ysq;

                int rs = n_1 - k;
                if (rs > 0)
                {
                    int j1 = j + 1;
                    int k1 = k + 1;
                    var w_sub = w.Subvector(j1, rs);
                    var L_sub = L_FF.Cols[k].Subvector(k1, rs);
                    w_sub.AddScaled(L_sub, -x_j/y);
                    if(gamma != 0)
                    {
                        // todo: combine both in one operation, e.g., by using submatrix and Matrix.ScaleAndAddProduct() rather than subvector
                        L_sub.Scale(sqrt_z / y);
                        L_sub.AddScaled(w_sub, sqrt_z * x_j / gamma);
                    }
                }
            }
        }

        // Update a Cholesky factor L_FF corresponding to the Cholesky decomposition of
        // the principal submatrix A_FF with index set F \in {1...n} for a full system matrix A based on all indices
        // by including the given index i in F, producing the new Cholesky factor L_{F'F'}, with F' = F u {i}, for
        // A_{F'F'}, that is, L_{F'F'} * L_{F'F'}^T = A_{F'F'}.
        //
        // The operation performed here is equivalent to the rank-1 update A_{F'F'} = A_{FF} + x*x^T with
        // x = L[i, :] being the i'th column of L.
        //
        // It is assumed that any rows and columns that have previously been removed via downdates from the full factorization L, with
        // A = L_L^T, have been left untouched in the factor.
        // Assumes that L has dimensions corresponding to a factorization for the full system A which includes all indices.
        public static void Update(in Matrix L_FF, int index)
        {
            int size = L_FF.MinDimension - 1;

            // Early return for last index
            if (index == size)
            {
                return;
            }

            int rows = size - index;
            int n = L_FF.MinDimension;
            var x = L_FF.Cols[index].Subvector(n - rows, rows);

            for(int k = 0; k < rows; ++k)
            {
                int kk = k + index + 1; // skip over zeros in L
                var l_kk = L_FF[kk, kk];
                var x_k = x[k];
                var r = math.sqrt(math.square(l_kk) + math.square(x_k));
                var c = r / l_kk;
                var s = x_k / l_kk;
                L_FF[kk, kk] = r;
                if(kk < size)
                {
                    int tail_size = size - kk;

                    // todo: combine most of the below in a single operation using submatrix instead of subvector and Matrix.ScaleAndAddProduct()

                    var L_kk_tail = L_FF.Cols[kk].Subvector(n - tail_size, tail_size);
                    var w_tail = x.Subvector(rows - tail_size, tail_size);
                    L_kk_tail.AddScaled(w_tail, s);
                    L_kk_tail.Scale(1/c);
                    w_tail.Scale(c);
                    w_tail.AddScaled(L_kk_tail, -s);
                }
            }
        }

        /// <summary>
        /// Computes the determinant from a Cholesky factor.
        /// </summary>
        /// <param name="L">The Cholesky factor.</param>
        /// <returns>The computed determinant of A.</returns>
        public static float ComputeDeterminantFromCholesky(in Matrix L)
        {
            var n = L.MinDimension;
            var determinant = 1.0f;
            for (int i = 0; i < n; i++)
            {
                determinant *= L[i, i];
            }
            return determinant;
        }
    }
}
