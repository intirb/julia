load("chol_blas.jl")
load("qr_blas.jl")

for np = [7 8]
	addprocs_local(np)
	print("Number of processors: ")
	println(nprocs())

	@everywhere load("pchol.jl")
	@everywhere load("pqr.jl")
	@everywhere libBLAS = dlopen("libopenblas")
	@everywhere load("blastmp.jl")

	for mat_size = [1000 2000 4000 8000 16000]
		print("Matrix size: ")
		println(mat_size)

		A = randn(mat_size,mat_size);
		B = A*A';

		chol(B);
		print("Julia Chol: ")
		tic()
		chol(B);
		toc()

		qr(A);
		print("Julia QR: ")
		tic()
		qr(A);
		toc()

		for n_blocks = [1 2 4 8]

			print("Number of blocks: ")
			println(n_blocks^2)
			
			tile_size = convert(Int64,mat_size/n_blocks)			

			chol_blas(B,tile_size)
			print("Serial BLAS Chol: ")
			tic()
			chol_blas(B,tile_size)
			toc()

			pchol(B,tile_size)
			print("Parallel BLAS Chol: ")
			tic()
			pchol(B,tile_size)
			toc()

			qr_blas(A,tile_size)
			print("Serial BLAS QR: ")
			tic()
			qr_blas(A,tile_size)
			toc()

			pqr(A,tile_size)
			print("Parallel BLAS QR: ")
			tic()
			pqr(A,tile_size)
			toc()

		end
	end
end
