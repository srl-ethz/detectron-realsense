# Archived!

This project has been moved into the [RAPTOR codebase](https://github.com/raptor-ethz/raptor) as a ROS2 package.


# detectron-realsense

This repository contains code for segmenting RGB frames from an Intel RealSense series camera using detectron2 and building a masked point cloud of the detected object. Furthermore, it contains code for sending compressed RGB and depth frames from the RealSense camera over a local network using imagezmq. This allows the analysis of captured frames on any connected computer in the same local network. Connection to other processes is enabled using ZeroMQ sockets. 

## Dependencies

This project depends on:

- detectron2
- OpenCV
- Open3D
- ZeroMQ
- imagezmq
- Protobuf
- pyrealsense2
- numpy
- pandas

It is recommended to install these dependencies to a [conda](https://www.anaconda.com/) environment. Particularly for detectron2, this may make the installation easier. For ARM devices, [miniforge](https://github.com/conda-forge/miniforge) is a suitable alternative to anaconda. 

## Usage and Setup

### Setup with streaming camera frames

This setup is suitable if you have a moving platform which cannot carry a computer powerful enough to achieve acceptable framerates running detectron2. In that case, you can get a small ARM computer that is connected to the RealSense camera, run the image streaming application on there and receive the images on a more powerful offboard computer. 

In that case, on the ARM computer connected to the camera, start streaming by running

```bash
python3 streamer_sender.py
```
and start processing on the offboard computer with

```bash
python3 main.py
```

To test image streaming, you can also start streaming on the ARM computer and run 

```bash 
python3 streamer_receiver.py
```
on the offboard computer. Make sure the IP address is set correctly in both programs as imagezmq uses IP-based forwarding without a discovery server that is commonly seen in ROS and similar systems.

### Integration into other systems

For integration into other systems, we provide [one example](https://github.com/erikbr01/Protocol-Buffer-Examples) for integration into a Fast DDS driven application. For integration, you only need to integrate ZeroMQ and Protobuf into your application and then you can connect this vision application with your target application. 

For questions, do not hesitate to reach out via a GitHub issue or per mail: [erbauer@ethz.ch](mailto:erbauer@ethz.ch)
