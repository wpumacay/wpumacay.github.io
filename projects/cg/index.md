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

The whole system and its features are depicted in the following figure.

![Img_architecture_features]({{site.url}}/assets/images/projects/img_cg_architecture_features.png)


## Implementation

### Rendering Engine

We implemented some basic features for the rendering engine, which are :

*	Abstracting OpenGL's functionality : Abstracted some funcionality like VBOs, VAOs, FBOs, materials, etc. into reusable parts in the engine.

*	Basic Lighting : Implemented Phong Shading and added support for multiple types of lights, like directional, spotlight and pointlight. The final simulator uses only directional lights as we are working on implementing deferred rendering for multiple shadow-enabled lights.

*	Shadows : We implemented shadow mapping to give the scene some more realism.

![Img_basic_features]({{site.url}}/assets/images/projects/img_cg_rendering_engine_basic_features.png)
<center><b><i>Basic Rendering Features</i></b></center>

*	Cameras : We implemented some basic cameras to use in the simulation.
![Gif_basic_features]({{site.url}}/assets/images/projects/gif_cg_cameras.gif)
<center><b><i>Testing cameras</i></b></center>

*	Debug drawings : We implemented a debug drawing system that uses batch rendering to draw various primitives, like lines, boxes, arrows, frustums, etc.

*	UI system : We also implemented a wrapper on top of the [Dear-ImGui](https://github.com/ocornut/imgui) library to make some UIs, which helps us for testing more quickly some functionality. Currently, we have abstracted the most basic widgets, and plan to wrap around some extra functionality that the library provides, like plotting time series.

![Img_basic_features]({{site.url}}/assets/images/projects/img_cg_rendering_engine_tools.png)
<center><b><i>UI and Debug draws</i></b></center>

### Wrapper around the Physics Engine

The Physics Engine works mostly with primitives that are simulated freely, or attached to each other by joints. We abstracted the necessary functionality and integrate it with the rendering engine.

![Gif_primitives_physics]({{site.url}}/assets/images/projects/gif_cg_primitives.gif)
<center><b><i>Bullet Collision shapes</i></b></center>

![Gif_primitives_joints]({{site.url}}/assets/images/projects/gif_cg_joints.gif)
<center><b><i>Bullet Joints</i></b></center>

### Entity Component System

We implemented an Entity Component system in order to integrate the different components more easily. The basic idea is depicted in the following picture.

![Img_entity_component_system]({{site.url}}/assets/images/projects/img_cg_entity_component_system.png)
<center><b><i>Entity-component overview</i></b></center>

Basically, an entity will be like a container for other objects that will handle some of its data ( Components ). The entity will serve as the bridge between the components, connecting parts like the physics transforms return from the physics engine, to the graphics components that are in charge of the rendering part, or hold the rendering information of the entity.

### Terrain Generation

The terrain generation is also key in the simulator. We implemented a basic terrain generator for the simulator which works similarly to the terrain generator in the reference project. 

We generate the terrain in a given direction, and make parts appear as they fit into the camera frustums. The terrain is integrated with the physics engine, supporting for now terrain generated by compounds of primitives. 

We are working on integrating the more general Heightfield collision shape of Bullet in order to have more general scenarios.

![Gif_entity_component_system]({{site.url}}/assets/images/projects/gif_cg_terrain.gif)
<center><b><i>Terrain Generation</i></b></center>