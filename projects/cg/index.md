---
layout: projects
---

# Design and Implementation of a simulation engine for research in locomotion.

![Intro]({{site.url}}/assets/images/projects/img_cg_scene_entities.png)

## Intro

This project focuses in the implementation from the ground-up of a simulation engine, based on the work done by [Michiel van de Panne](https://www.cs.ubc.ca/~van/) and its group at the University of British Columbia; in specific, the project [**Terrain-Adaptive Locomotion Skills Using Deep Reinforcement Learning**](https://xbpeng.github.io/projects/DeepTerrainRL/index.html), by [Xue Bin Peng](https://xbpeng.github.io/).

## Objectives

We designed the architecture of the modules to be used for this simulator with the aim to reproduce the results of the project mentioned before, as well as use the resulting engine for some other research purposes. 

We have two main goals in this project :

*	Make a rendering engine to visualize the simulation, and reuse it in other projects.
*	Make a physics wrapper of the Bullet3 physics engine to handle physics simulations.

## About the reference project

As described earlier, we based our project on the **Terrain-Adaptive Locomotion Skill Using Deep Reinforcement Learning**. This paper focuses on training controllers using Deep Reinforcement Learning in order to achieve locomotion for simulated characters. The project consists of a simulation engine built by the authors, which they used in several projects related to locomotion, like [this](https://xbpeng.github.io/projects/TerrainRL/index.html), [this](https://xbpeng.github.io/projects/TerrainRL/index.html), and more recently [this](https://xbpeng.github.io/projects/ActionSpace/index.html).

Below you can see the demo provided by the authors.

#### Image here!!!!

## Design Architecture 

There are three main modules implemented for this project, which are :

*	**Graphics Engine** : To render the simulated scene.
*	**Physics wrapper** : To interface with the Bullet3 physics engine.
*	**Controller wrapper** : To interface with the project's controllers code.

![Img_architecture]({{site.url}}/assets/images/projects/img_cg_architecture.png)

### Graphics Engine

A graphics engine is needed to render the simulated scene. The original project uses the old OpenGL's rendering pipeline in a inmediate render mode, by means of glBegin and glEnd. 

We implemented a basic rendering engine to handle the rendering process using the modern OpenGL pipeline ( OpenGL 3 and GLSL 330 core ).

In this stage we had to abstract the functionality of the OpenGL pipeline to be able to use it easily, as well as implementing some necessary visual features like lighting, shadows, debug drawings, etc.

### Physics wrapper

The authors used the [Bullet3](http://bulletphysics.org/wordpress/) physics engine, which is a popular engine used for collision detection.

We implemented a simple wrapper on top of Bullet to integrate it with the rendering engine. This is achieved by means of an Entity - Component architecture that allows us to integrate different components into the whole architecture.

### Controller Wrapper

This last module is in charge of integrating the code from the authors, which handles the torques calculation done by controllers, torques that are input to the joints of the simulated character. 

The main work of the authors in this paper was the implementation of these controllers, which are various modules on top of [Caffe](http://caffe.berkeleyvision.org/) that take some features from the simulation and calculate the necessary torques for appropiate locomotion.

### Overall design

The whole system and its features is depicted in the following figure.

![Img_architecture_features]({{site.url}}/assets/images/projects/img_cg_architecture_features.png)


## Implementation





