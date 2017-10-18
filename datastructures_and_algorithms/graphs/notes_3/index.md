---
layout: notes_interactive_2d
---

# Weighted Graphs

In the previous section we learnt about unweighted graphs and talked about its applications and some
algorithms we could use to solve some particular problems, like traversals, find connected components, etc.
There is actually quite some problems you can model with unweighted graphs, but once you add weights in the mix
you get a lot more of modelling power.

In this section we will focus on weighted graphs and weighted graph algorithms and its applications. They arise naturally
in various applications, like finding shortests routes in a traffic network, finding the fastest route for a packet in the internet, etc., just to name a few.

## Sample Weight Graph

Let's start by presenting a simple weight graph and how to implement them. The next figure shows an example
of an undirected weighted graph.

TODO: app here for UWG

The way we can extend our **Graph** data type is by encoding the the weights in our **Edge** objects. In the following snippet
you can see how we augmented our Edge class with a weight field, so that we can encode weights into them. Remember, we are still using the Adjacency List implementation of our graph, so each edge in the adjacency list of a given node now has a 
weight associated with it.

TODO: code snippet here

The next figure shows an example of the directed weighted case, which is similar to the previous one, just with directed edges in the mix.

TODO: app here for DWG

Ok, so now that we can represent weights, let's talk about some problems and algorithms that arise when dealing with weighted graphs.


## Minimum Spanning trees

We will first discuss the minimum spanning tree problem. So, let's define what is a spanning tree and why it might be useful.

### Spanning trees

A spanning tree of an undirected graph $$ G=(V,E) $$ is a subgraph **T** that is a __tree__ and contains every vertex of G.

In the following figure we have a simple undirected weighted graph and some cases of spanning trees for that graph.

{% include container_2d_graphs_mst_st1.html %}

As you can see, there are various spanning trees, two of them we have formed using simple graph traversals.

### Minimum spanning tree

As we saw before, there can be lots of spanning trees of a given graph. If the graph has weights, then we certanly would be interested in the spanning tree which has
minimum weight, that is, a subgraph which is a spanning tree in which the sum of all weights for its edges is minimum. This give raise to the minimum spanning tree problem, which
is a very interesting one.

So, the idea is to find the ***Minimum Spanning Tree*** for a given undirected weighted graph $$G=(V,E)$$. This can be done in a kind of general way following an incremental greedy approach, as follows:

\\[
    T = MST(G) \textit{ can be build incrementally by adding safe edges.}
\\]

A **safe edge** for a current built spanning tree **T** is an edge $$e = (u,v) \in E$$ that such that $$ T \cup \lbrace e \rbrace $$ is part of a **MST**

By using this idea, we can write down the following generic algorithm to build a **MST**.

```
Generic_MST( G )
    Let T = {}
    while T doesn't form a spanning tree
        find a safe edge ( u, v ) that is safe for T
        add ( u, v ) to T
```

This algorithm is of the greedy kind, and the following algorithms that we will discuss will build a MST using this idea.

### Prim's algorithm

{% include container_2d_graphs_mst_prim.html %}


## Shortest paths algorithms

In this section we will discuss about how to traverse a graph.

fun foo

foo bar

### Shortests paths - approach

fun foo



foo bar

### Dijkstra







### Bellman Ford






### All pair shortest paths - Floyd Warshall

