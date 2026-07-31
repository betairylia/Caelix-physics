using NUnit.Framework;
using Unity.Numerics.Linear.Dense.Primitives;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Numerics.Memory;
using Unity.PerformanceTesting;
using Unity.Physics.Tests;
using UnityEngine;

namespace Unity.Numerics.Linear.Tests
{
    public unsafe partial class TestDecompositions
    {
        [BurstCompile(CompileSynchronously = true)]
        [GenerateTestsForBurstCompatibility]
        struct CholeskyDecompAndReconstructJob : IJob
        {
            [ReadOnly] public Matrix M;

            public void Execute()
            {
                using var L = M.Heap.Matrix(M.NumRows, M.NumCols);
                Cholesky.Factor(M, L, out var singularRow);
                Assert.IsTrue(singularRow == -1);

                Cholesky.Factor(M, out singularRow);
                Assert.IsTrue(singularRow == -1);

                // clean up the result
                for (int i = 0; i < M.MinDimension; i++)
                {
                    M.Rows[i].Clear(i + 1);
                }

                // make sure in-place version and out-of-place version are the same
                float maxDiff = MaxAbsoluteDiff(M, L);
                Assert.IsTrue(maxDiff < 1.0e-5f);

                var determinant = Cholesky.ComputeDeterminantFromCholesky(M);
                Assert.IsTrue(determinant > 0);

                // Compute L L^t
                M.MultiplyByTriangular(Side.Right, TriangularType.Lower, Op.Transpose, DiagonalType.Explicit, 1.0f, M);
            }
        }

        static void TestCholesky(in MemoryManager heap, int m, int n, out Matrix A, out Matrix Cholesky)
        {
            Cholesky = heap.Matrix(m, n);
            var generateRandomMatrixJob = new GenerateRandomSymmetricPositiveDefiniteMatrixJob
            {
                Matrix = Cholesky
            };

            generateRandomMatrixJob.Run();

            A = Cholesky.Copy();

            var choleskyJob = new CholeskyDecompAndReconstructJob
            {
                M = A,
            };
            choleskyJob.Run();
        }

        [Test]
        public void Cholesky_Job_Decompose10x10_CompareProductToOriginal_CheckDeterminant()
        {
            using (var heap = MemoryManager.Create(16384, Allocator.Persistent))
            {
                using var M = Matrix.Create(heap, 10, 10);
                var job = new GenerateRandomSymmetricPositiveDefiniteMatrixJob
                {
                    Matrix = M
                };
                job.Run();

                using var original = M.Copy();

                var choleskyJob = new CholeskyDecompAndReconstructJob
                {
                    M = M,
                };
                choleskyJob.Run();

                float maxDiff = MaxAbsoluteDiff(original, M);
                Assert.IsTrue(maxDiff < 1.0e-5f);
            }
        }

        [Test]
        public void Cholesky_Job_DecomposeRandom500x500_CompareProductToOriginal()
        {
            if (!BurstHelper.IsBurstEnabled())
            {
                Assert.Ignore("This test is too slow with Burst disabled and times out.");
            }

            using (var heap = new MemoryManager(512 * 512 * 128, Allocator.Temp))
            {
                TestCholesky(heap, 500, 500, out var A, out var Cholesky);

                float maxDiff = MaxAbsoluteDiff(A, Cholesky);
                Assert.IsTrue(maxDiff < 5.0e-5f);
                Cholesky.Dispose();
                A.Dispose();
            }
        }

        [Test]
        // unit test for updating and downdating of a Cholesky factorization.
        public void Cholesky_DowndateUpdate([Values(4, 10, 100)] int numRows)
        {
            using var heap = new MemoryManager(Allocator.Persistent);

            using var tight = new NativeList<int>(Allocator.Temp);
            using var free = new NativeList<int>(Allocator.Temp);
            int n = numRows;
            CreateTightAndFreeIndexSets(n, tight, free);

            using var A_full = heap.Matrix(n, n);
            using var L_full = heap.Matrix(n, n);

            // Create a random problem
            new Random.Random(42).GenerateRandomSymmetricPositiveDefiniteMatrix(heap, A_full);

            // Initial decomposition of full matrix
            // Compute the Cholesky factorization of the full matrix A = L3*L3^T
            Cholesky.Factor(A_full, L_full, out var singularRow);
            Assert.IsTrue(singularRow == -1);

            // Do a complete Cholesky factorization for the L matrix given the 'free' set.
            // Create principal submatrix A_FF of A corresponding to the free variables.
            using var A_FF = heap.Matrix(free.Length, free.Length);
            using var L_FF = heap.Matrix(free.Length, free.Length);
            A_full.CreateSubmatrix(A_FF, free.AsArray(), free.Length, free.AsArray(), free.Length);
            // Compute Cholesky factorization of A_FF = L1*L1^T
            Cholesky.Factor(A_FF, L_FF, out singularRow);
            Assert.IsTrue(singularRow == -1);

            // Downdate a full factorization to remove the tight variables, and compare the result
            // with a from scratch factorization for only the free set.

            using var L_downdated = L_full.Copy();
            var currentIndexSet = new NativeHashSet<int>(n, Allocator.Temp);
            for (int i = 0; i < n; ++i)
            {
                currentIndexSet.Add(i);
            }

            foreach (var index in tight)
            {
                Cholesky.Downdate(L_downdated, index);
                currentIndexSet.Remove(index);

                using var tmp = heap.Matrix(currentIndexSet.Count, currentIndexSet.Count);
                L_downdated.CreateSubmatrix(tmp,
                    currentIndexSet.ToNativeArray(Allocator.Temp), currentIndexSet.Count,
                    currentIndexSet.ToNativeArray(Allocator.Temp), currentIndexSet.Count);
                // Debug.Log(new MatrixDebugView(tmp).MatlabString);
            }
            // @todo: try this but with a smaller matrix and see the effect of the downdate

            // compare downdated factorization with the from-scratch factorization of A_FF
            // need to extract submatrix of L_downdated corresponding to the free variables for easy comparison
            using var L_downdated_FF = heap.Matrix(free.Length, free.Length);
            L_downdated.CreateSubmatrix(L_downdated_FF, free.AsArray(), free.Length, free.AsArray(), free.Length);
            var maxDiff = MaxAbsoluteDiff(L_downdated_FF, L_FF);
            Assert.IsTrue(maxDiff < 1.0e-5f);


#if false   // @todo: that doesn't seem to work yet... Some numerical issues?
            // Update the same factorization to include the tight set again and compare the result with the factorization for the full A matrix:
            // We will have to update in reverse order to be able to undo the downdating process properly.
            // @todo: confirm above!
            using var L_updated = L_downdated.Copy();
            for (int i = tight.Length - 1; i >= 0; --i)
            {
                var index = tight[i];
                Cholesky.Update(L_updated, index);

                currentIndexSet.Add(index);
                using var tmp = heap.Matrix(currentIndexSet.Count, currentIndexSet.Count);
                L_updated.CreateSubmatrix(tmp,
                    currentIndexSet.ToNativeArray(Allocator.Temp), currentIndexSet.Count,
                    currentIndexSet.ToNativeArray(Allocator.Temp), currentIndexSet.Count);
                Debug.Log(new MatrixDebugView(tmp).MatlabString);
            }

            // compare updated factorization with the full factorization
            maxDiff = MaxAbsoluteDiff(L_updated, L_full);
            Assert.IsTrue(maxDiff < 1.0e-5f);
#endif
        }

        static void CreateTightAndFreeIndexSets(int n, NativeList<int> tight, NativeList<int> free)
        {
            tight.Clear();
            free.Clear();

            var random = new Random.Random(42);
            for (int i = 0; i < n; ++i)
            {
                var p = random.NextUniform();
                if (p < 0.5)
                {
                    free.Add(i);
                }
                else
                {
                    tight.Add(i);
                }
            }
        }

#if PHYSICS_ENABLE_PERF_TESTS

        [BurstCompile(CompileSynchronously = true)]
        [GenerateTestsForBurstCompatibility]
        struct CholeskyFactorizeJob : IJob
        {
            [ReadOnly] public Matrix M;
            public Matrix MFactor;

            public void Execute()
            {
                M.CopyTo(MFactor);
                Cholesky.Factor(MFactor, out var singularRow);
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        [GenerateTestsForBurstCompatibility]
        struct CholeskyFactorizeAndSolveJob : IJob
        {
            [ReadOnly] public Matrix M;
            public Matrix MFactor;
            public Matrix RHS;

            public void Execute()
            {
                M.CopyTo(MFactor);
                Cholesky.Factor(MFactor, out var singularRow);
                Assert.IsTrue(singularRow == -1);

                // After factorization of a matrix M = LL^t with Cholesky.Factor(M, ...), M contains L in its
                // lower triangular block including the diagonal.

                // Solve linear system LL^t * x = rhs via forward and backward substitution as follows:
                //      1. Define y := L^t * x
                //      2. Solve L * y = rhs for y
                //      3. Then, solve L^t * x = y for lambda

                // Solve L * y = rhs for y
                // Note: RHS.Col[0] = rhs
                RHS.SolveGeneralizedTriangular(Side.Left, TriangularType.Lower, Op.None, DiagonalType.Explicit,
                    1.0f, MFactor);
                // Note: at this point, RHS contains y

                // Solve L^t * x = y for x
                RHS.SolveGeneralizedTriangular(Side.Left, TriangularType.Lower, Op.Transpose, DiagonalType.Explicit,
                    1.0f, MFactor);
            }
        }

        [Test, Performance]
        public void Cholesky_DecomposeRandom_Performance([Values(50, 100, 200, 300, 400, 500, 750, 1000)] int dimension)
        {
            if (dimension > 10 && !BurstHelper.IsBurstEnabled())
            {
                Assert.Ignore("This test variant is time consuming and is therefore only run with Burst enabled.");
            }

            using (var heap = new MemoryManager(1024 * 1024 * 128, Allocator.Persistent))
            {
                using var M = heap.Matrix(dimension, dimension);
                using var LLT = heap.Matrix(dimension, dimension);
                var generateMatrix = new GenerateRandomSymmetricPositiveDefiniteMatrixJob
                {
                    Matrix = M
                };
                generateMatrix.Run();

                var job = new CholeskyFactorizeJob
                {
                    M = M,
                    MFactor = LLT
                };

                Measure.Method(() => job.Run())
                    .WarmupCount(1)
                    .MeasurementCount(10)
                    .Run();
            }
        }

        [Test, Performance]
        public void Cholesky_DecomposeRandomAndSolve_Performance([Values(50, 100, 200, 300, 400, 500, 750, 1000)] int dimension)
        {
            if (dimension > 10 && !BurstHelper.IsBurstEnabled())
            {
                Assert.Ignore("This test variant is time consuming and is therefore only run with Burst enabled.");
            }

            using (var heap = new MemoryManager(1024 * 1024 * 128, Allocator.Persistent))
            {
                using var M = heap.Matrix(dimension, dimension);
                using var LLT = heap.Matrix(dimension, dimension);
                using var RHS = heap.Matrix(dimension, 1); // Right-hand side vector
                RHS.Clear(); // simply set RHS to zero for linear system solve: M * x = rhs = 0

                var generateMatrix = new GenerateRandomSymmetricPositiveDefiniteMatrixJob
                {
                    Matrix = M
                };
                generateMatrix.Run();

                var job = new CholeskyFactorizeAndSolveJob
                {
                    M = M,
                    MFactor = LLT
                };

                Measure.Method(() => job.Run())
                    .WarmupCount(1)
                    .MeasurementCount(10)
                    .Run();
            }
        }

#endif
    }
}
