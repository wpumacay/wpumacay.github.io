---
layout: notes
---

# [](#header-1) LU decomposition

In this section we will see how to solve a linear system of equations via the LU decomposition/ factorization method, which basically tries to transform a matrix $$A$$ into a product $$LU$$.

First, why should we care about these factorization?. Well, by factorizing the system $$ Ax=b $$ into the system $$ LUx = b $$ we can make the solution process way more simpler.
We just have to make a two-step process to calculate the solution $$x$$. First, solve the system $$Ly = b$$, which is simpler because the $$L$$ matrix is lower triangular, which is equivalent to applying a kind of backsubstitution, like we did in the final step of gaussian elimination. Then we end up with a system $$Ux = y$$, which we solve by direct backsubstitution, which is also simple. 

Now, the problem is how to find this factorization, which is what we are going to study in this section and the next.

The first approach is by using elementary matrices, which make the process quite similar to gaussian elimination. What we aim is to change the matrix $$A$$ iteratively by making zeros below the main diagonal, thus converting the matrix $$A$$ into an upper triangular matrix. So, let's check the primitive, namely elementary matrices.

### Elementary matrices

These matrices are some special kind of matrices that we will use to make zeros in our target matrix. They have the form of a identity matrix, with the exception that they have some non-zero elements below the diagonal in a certain column, like the following matrix.

\\[ 
	\begin{bmatrix}
		1 	& 	0 	& 	0 	 	& 	0 	& 	0 \\\
		0 	& 	1 	& 	0 	 	& 	0	& 	0 \\\
		0 	& 	0 	& 	1 	 	& 	0 	& 	0 \\\
		0 	& 	0 	& 	m_{4,3} & 	1 	& 	0 \\\
		0 	& 	0 	& 	m_{5,3} & 	0 	&  	1
	\end{bmatrix}
\\]

As we see, these matrix is essentially an identity matrix, but with zeros below the diagonal element in the 3rd column.

More generally, a lower elementary matrix can be defined as follows :

\\[
	E = I_{k} + m_{k} e_{k}^{T}
\\]

Where $$e_{k}$$ is the kth vector of the canonical base in $$R^{n}$$, which is like ( for example, $$e_{2}$$ in $$R^{5}$$ ) :

\\[
	e_{2} = 
	\begin{bmatrix}
		0 \\\ 1 \\\ 0 \\\ 0 \\\ 0
	\end{bmatrix}
\\]

The vector $$m_{k}$$ is a special kind of column vector in $$R^{n}$$, which has all elements zero from the first to the kth element, like this ( for $$m_{3}$$ in $$R^{5}$$):

\\[
	m_{k} = 
	\begin{bmatrix}
		0 \\\ 0 \\\ 0 \\\ m_{4} \\\ m_{5}
	\end{bmatrix}
\\]

The matrix above was actually formed by $$ E = I_{5} + m_{3} e_{3}^{T} $$, which I recommend checking out by computing it.

As we can see, an elementary matrix is formed by its $$m_{k}$$ vector of elements, which in the case of the $$LU$$ factorization is known as vector of **multipliers**.

Let's see how to compute the $$LU$$ factorization by using these elementary matrices.

### LU factorization using elementary matrices

This process is similar to Gaussian Elimination. What we want is to make zeros below the elements of the diagonal to get a resulting upper triangular system.

Unlike Gaussian elimination, we also need to compute a lower triangular matrix $$L$$ which at first sight seems like a completely different approach. The key is to note that while executing Gaussian elimination, we are actually computing some elements that are part of the lower triangular matrix that we want. These elements are the multipliers that we use in order to make zeros below the diagonal. These multipliers are the actual ones we are going to use to compute our elementary matrices.

Recall that an elementary matrix is defined by its vector $$m_{k}$$, which in the context of LU factorization is actually the vector of multipliers in the kth iteration of the gaussian elimination process.

Once we have these multipliers, we can form an elementary matrix such that, if multiplies the matrix $$A$$ to the left will make zeros below the kth diagonal element in the kth column.

Ok, ok, sorry for too many words, let's see a concrete example of how to form these matrices.

Let's use the following matrix :

\\[
	A = 
	\begin{bmatrix}
		2	&	2	&	3 \\\
		4	&	5	&	6 \\\
		1	&	2	&	4
	\end{bmatrix}
\\]