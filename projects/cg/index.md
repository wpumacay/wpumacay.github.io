---
layout: projects
---

# On the Architecture and Implementation of a simulation engine for research in locomotion.

#### Image here!!!!

## Intro

This project focuses in the implementation from the ground-up of a simulation engine, based on the work done by [Michiel van de Panne](https://www.cs.ubc.ca/~van/) and its group at the University of British Columbia; in specific, the project [**Terrain-Adaptive Locomotion Skills Using Deep Reinforcement Learning**](https://xbpeng.github.io/projects/DeepTerrainRL/index.html), by [Xue Bin Peng](https://xbpeng.github.io/).

We designed the architecture of the modules to be used for this simulator with the aim to reproduce the results of the project mentioned before, as well as use the resulting engine for some other research purposes.

## About the reference project

As described earlier, we based our project on the **Terrain-Adaptive Locomotion Skill Using Deep Reinforcement Learning**. This paper focuses on training controllers using Deep Reinforcement Learning in order to achieve locomotion for simulated characters. The project consists of a simulation engine built by the authors, which they used in several projects related to locomotion, like [this](https://xbpeng.github.io/projects/TerrainRL/index.html), [this](https://xbpeng.github.io/projects/TerrainRL/index.html), and more recently [this](https://xbpeng.github.io/projects/ActionSpace/index.html).

Below you can see their provided github demo.

#### Image here!!!!

## Architectural design

There are three main modules implemented for this project, which are :

*	**Graphics Engine** : To render the simulated scene.
*	**Physics wrapper** : To interface with the Bullet3 physics engine.
*	**Controller wrapper** : To interface with the project's controllers code.


### Graphics Engine

A graphics engine is needed to render the simulated scene. The original project uses the old OpenGL's rendering pipeline in a inmediate render mode, by means of glBegin and glEnd. 

We implemented a basic rendering engine to handle the rendering process using the modern OpenGL pipeline ( OpenGL 3 and GLSL 330 core ).

In this stage we have to abstract the functionality of the OpenGL pipeline to be able to use it easily, as well as implementing features like