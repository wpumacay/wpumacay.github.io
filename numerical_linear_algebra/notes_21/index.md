---
layout: notes
---

# Gradient method


The iteration is as follows :

\\[
	\textbf{ Gradient iteration }\\\
	\alpha_{k} = \frac{r_{k}^{T}r_{k}}{p_{k}^{T}Ap_{k}}\\\
	x_{k+1} = x_{k} + \alpha_{k} p_{k} \\\
	r_{k+1} = r_{k} - \alpha_{k} A p_{k} \\\
	\beta_{k} = \frac{r_{k+1}^{T}r_{k+1}}{r_{k}^{T}r_{k}} \\\
	p_{k+1} = r_{k+1} + \beta_{k} p_{k}
\\]
