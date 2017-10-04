---
layout: notes
---

# Conjugate Gradient method

### Iteration step

The iteration for the conjugate gradient is quite similar to the basic gradient method. The only difference are the direction vectors used 
to update the solution vector in the iteration step. Instead of the residual, we compute a conjugate direction based on the residual.

The iteration is as follows :

\\[
	\textbf{ Conjugate Gradient iteration }\\\
	\alpha_{k} = \frac{r_{k}^{T}r_{k}}{p_{k}^{T}Ap_{k}}\\\
	x_{k+1} = x_{k} + \alpha_{k} p_{k} \\\
	r_{k+1} = r_{k} - \alpha_{k} A p_{k} \\\
	\beta_{k} = \frac{r_{k+1}^{T}r_{k+1}}{r_{k}^{T}r_{k}} \\\
	p_{k+1} = r_{k+1} + \beta_{k} p_{k}
\\]
