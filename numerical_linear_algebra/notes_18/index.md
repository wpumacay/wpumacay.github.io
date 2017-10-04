---
layout: notes
---


# Jacobi and Gauss Seidel methods

In this section we will start studying iterative methods for solving systems of linear equations.

Contrary to direct methods, iterative methods do not find the exact solution in an exact number of steps ( note: exception being the conjugate gradient method ), but instead in an infinite number of steps. What we want is usually a good enough aproximation of the exact solution, which can be obtained in some iterations.

The advantage of iterative methods over direct ones is that they deal better with sparse systems, being special iterative methods that can deal with these systems and compute a solution efficiently and with less computation than direct methods.

### Iterative step

The general iterative step to calculate a solution for the system $$Ax=b$$ is as follows :

\\[
	x^{(k+1)} = B x^{(k)} + f
\\]

With *B* and *f* calculated from *A* and *b*, usually by partitioning the matrix *A*.

Of course, these iterative step has to be derived from the original system and yield the solution of this system, not any other system. That's why, when the system converges, $$x^{(k+1)}$$ should be the same as $$x^{(k)}$$. It's like a kind of "steady state" solution.

This solution should also be equal to $$A^{-1}b$$, so by comparing the iterative step in the steady state, and the exact solution, we can get the following :

\\[
	x = B x + f \\\
	\rightarrow 
	(I - B)x = f \\\
	\rightarrow
	(I - B)A^{-1}b = f \\\
\\]

So we say that a system is consistent ( if converges, converges to the solution of our system ) if satisfies this:

\\[
	f = (I - B)A^{-1}b
\\]

Of course, consistency does not suffice to guarantee convergence for any initial guess, which is what we aim for.

We will talk about the convergence of an iterative method in the next sections. For now, rest assure that there are checks that we can do to ensure that our iterative method will converge to the exact solution for any initial guess.

### Jacobi method

As mentioned earlier, we can obtain the matrix *B* by splitting the matrix *A*. According to the splitting, we get different methods, being the Jacobi method the one corresponding to the splitting of the matrix in the following form :

\\[
	A = D + ( L + U ) = D + "LU"
\\]

Where we split the matrix *A* into *D* (diagonal part of *A*) and *( L + U )* ( non diagonal part of *A* ). This gives us the following iterative method.

\\[
	\textbf{ Jacobi iteration } \\\
	x^{k+1} = B_{J} x^{k} + b_{J} \\\
	B_{J} = -D^{-1}( L + U ) \\\
	b_{J} = D^{-1}b
\\]

You can check that these relations are satisfied. Just apply the splitting and work your way to the general iterative step form, and you will find out these relations.

A more compute-friendly expression can be obtained, if you don't want to code the matrix multiplications and generate these iterative matrices at all. Let's just start with the standard form :

\\[
	x^{k+1} = B_{J} x^{k} + b_{J} = -D^{-1}( L + U )x^{k} + D^{-1}b
\\]

If we factorize $$D^{-1}$$ in the right hand side, we get :

\\[
	x^{k+1} = D^{-1}( b - ( L + U )x^{k} )
\\]

If we focus in the *ith* element of the vector $$x^{k+1}$$ we can get the following expresion :

\\[
	x_{i}^{(k+1)} = \frac{1}{a_{ii}} ( b_{i} - \sum_{j \neq i} a_{ij} x_{j}^{(k)} )
\\]

Which can be implemented without the need of computing the corresponding matrix and vector of the iterative step. Also if you don't have a matrix library at your disposal :D .

### Worked example on the Jacobi method

TODO: Write an example

### MATLAB, Python and C/C++ implementation

TODO: Write the implementation for the appropiate cases, or maybe make another section in which you deal
with implementation details.

### Gauss Seidel method

The Gauss Seidel method is obtained by the following splitting : 

\\[
	A = ( D + L ) + U = "DL" + U
\\]

Now the preconditioning matrix *P* is $( D + L )$, and the iterative step can be expressed as follows :

\\[
	\textbf{ Gauss Seidel iteration } \\\
	x^{k+1} = B_{GS} x^{k} + b_{GS} \\\
	B_{GS} = -( D + L )^{-1} U \\\
	b_{GS} = ( D + L )^{-1}b
\\]

Again, by working with the expressions, 