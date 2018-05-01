---
layout: tutorials
---

# ROS - Basics demo

We have already some tools to start building a simple ROS application. We will start by making a simple app based on the **Udacity-RoboND** demo called *SimpleArm*.

## Intro

TODO: 

## Setup

TODO: 

## Understanding the package

First, let's check which nodes we are dealing with when we launch the application. A simple *rosnode list* and *rostopic list* will help us find out what is going on inside the package.

    $ rosnode list
        /gazebo
        /joint_state_publisher
        /robot_state_publisher
        /rosout
        /simple_arm/spawner

Well, we have some nodes running. Still, by the names it seems that the */simple_arm/spawner* node just was in charge of spawning the 2dof arm. After digging a bit inside the package, it seems that the */joint_state_publisher* and */robot_state_publisher* nodes are just there to do not much. If you comment them in the *robot_description.xml* file, and rerun the application everything works as it should. Maybe they were there for testing purposes, as the */joint_state_publisher* when opened in *gui* mode, gives a nice GUI to play with the joint angles. If you enable GUI mode for that node, you will get a GUI that does nothing to the robot. It is publishing to the */joint_states* topic, and you can even play with the GUI and echo the topic and see that indeed it changes it, but does nothing ( at least visually ) to the robot, as the robot joints are controller by **JointPositionController**, found in the *controllers.yaml* file, inside the *controllers* folder.

Let's look at the topics to see what else does the package provides.

    $ rostopic list
        /clock
        /gazebo/link_states
        /gazebo/model_states
        /gazebo/parameter_descriptions
        /gazebo/parameter_updates
        /gazebo/set_link_state
        /gazebo/set_model_state
        /joint_states
        /rgb_camera/camera_info
        /rgb_camera/image_raw
        /rgb_camera/image_raw/compressed
        /rgb_camera/image_raw/compressed/parameter_descriptions
        /rgb_camera/image_raw/compressed/parameter_updates
        /rgb_camera/image_raw/compressedDepth
        /rgb_camera/image_raw/compressedDepth/parameter_descriptions
        /rgb_camera/image_raw/compressedDepth/parameter_updates
        /rgb_camera/image_raw/theora
        /rgb_camera/image_raw/theora/parameter_descriptions
        /rgb_camera/image_raw/theora/parameter_updates
        /rgb_camera/parameter_descriptions
        /rgb_camera/parameter_updates
        /rosout
        /rosout_agg
        /simple_arm/joint_1_position_controller/command
        /simple_arm/joint_1_position_controller/pid/parameter_descriptions
        /simple_arm/joint_1_position_controller/pid/parameter_updates
        /simple_arm/joint_1_position_controller/state
        /simple_arm/joint_2_position_controller/command
        /simple_arm/joint_2_position_controller/pid/parameter_descriptions
        /simple_arm/joint_2_position_controller/pid/parameter_updates
        /simple_arm/joint_2_position_controller/state
        /simple_arm/joint_states
        /tf
        /tf_static

Quite some topics :D. As we can see the *simple_arm* package has some topics it exposes, two of which we will be publishing to : **/joint_1_position_controller/command** and **/joint_2_position_controller/command** topics. When publishing to them, we will be sending joint commands to the controllers, and then they will do their job and set the position to the desired reference.

Just for fun, let's check the graph of the application.

    $ rosrun rqt_graph rqt_graph

![Img_graph]({{site.url}}/assets/images/tutorials/ros/img_tutorials_ros_basics_demo_graph.png)

Just to be clear, I have the **arm_mover** node running ( which is the one we will be making from scratch ) in order to see the topic it's publishing to. As you can see, it is publishing to the command topics mentioned earlier; these are then sent to the simulator and then these are published to the */simple_arm/joint_states* topic. We will make the **simple_mover** and **arm_mover** nodes a bit more complex than the original nodes made in the udacity course, by adding a simple GUI on top of Qt to handle the joints.

## First basic node - Simple Mover

