---
layout: notes
---

# [](#header-1) Intro

Well, linear algebra it's quite broad. There are lots of information out there on the internet. I recommend watching
Gilbert Strang's lectures on linear algebra ( MIT ocw ) to get the basics.

These notes are just a summary to help you remember some concepts.

# Vector spaces and subspaces

The concept of a vector space is key in the study of lienar algebra. It can be defined as follows.

### Vector space

A Vector Space **V** over a field **F** ( real numbers, complex numbers, etc ) is a set of elements called **vectors** in which there are two defined operations, namely addition and scalar-vector multiplication, which satisfy the following 8 axioms :

1. 	Associativity :  
	
	$$ u + ( v + w ) = ( u + v ) + w, \forall u,v,w \in V $$

2. 	Commutativity :  

	$$ u + v = v + u, \forall u,v \in V $$

3. 	Existance of identity element for vector addition : 

	$$ \exists \theta \in V \textit{ such that } \theta + u = u, \forall u \in V $$

4.  Inverse element existance :
	
	$$ \forall v \in V, \exists -v \in V \textit{ such that } v + ( -v ) = \theta $$

5.  "Associativity" in scalar-vector multiplication :

	$$ a(bv) = (ab)v , \forall a,b \in F, \forall v \in V $$

6.  Existance of identity element for scalar-vector multiplication :

	$$ \exists i \in F \texit{ such that } 1(v) = v, \forall v \in V $$

7.	Distributivity of scalar-vector multiplication over vector addition :

	$$ a( v + w ) = av + aw, \forall a \in F, \forall v,w \in V $$

8.	Distributivity of scalar-vector multiplication over field addition :

	$$ ( a + b )v = av + bv, \forall a,b \in F, \forall v \in V $$

I've extracted these from wikipedia, there is more info about the definition and other stuff. What is important to understand is that a vector space is a kind of special type of set that has these fancy properties associated with it. This kind of special set is formed by a general kind of element, called **vectors**, which must satisfy this properties. We haven't defined a specific kind of vectors, they could be anything, and that is the power of this definition. We could be dealing with the common kind of vectors we usually deal in 3 dimensions when using cartesian coordinates and doing some calculus, but they could also be polynomials, functions, matrices, solutions to differential equations, etc. As long as they satisfy these axioms, they form a vector space and we can then work in a framework which give us lots of good tools, called Linear Algebra.

Let's see some examples of vector spaces to make the denition more clear.

### Some examples of vector spaces

TODO: Explain some sample vector spaces from [here]( https://en.wikibooks.org/wiki/Linear_Algebra/Definition_and_Examples_of_Vector_Spaces ) and from my own notes.

### Subspaces

TODO: Define subspaces and its properties.