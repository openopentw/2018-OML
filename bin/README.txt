There are total 4 matlab files in src, and you should put the
`libsvmread.mexa64` and `kddb` into the same path (src/) as the 4 matlab
files.

Or, additionally, you can also manually modify the path in the
`experiment_*.m` file.

For example:

	- Report.pdf
	- src
		- experiment_GD.m
		- experiment_NM.m
		- GD.m
		- NM.m

		// put below files in the same path

		- libsvmread.mexa64
		- kddb

To run Gradient Descent Method, run `experiment_GD.m` .

To run Newton Method, run `experiment_NM.m` .

The result will be the four tuple
	[iters, iters of alpha, norm of gradient, f(w_iter)]
for each iteration.
