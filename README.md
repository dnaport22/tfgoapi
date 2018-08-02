# TFGraphApi
TensorFlow Graph Api to execute tensflow based pre-trained models. This project already has 2 models, i.e. Face Detection and Object Detection.

This project is a part of Deep Fashion Project and is intended to work along with all other smart micro-services under the project.

# How to run
## Prerequisites
 - [Tensorflow](https://www.tensorflow.org/install/install_c)
 - [Golang](https://golang.org/)

Once the all of the requirements are satisfied, we can start playing with the api. In order to get the codebase and execute pre-complied binary, follow the steps:

```bash
> cd DESIRED/PROJECT/DIR
> git clone ... tfGraphApi
> cd tfGraphApi/bin
> ./main -u objectDetection -img grace_hopper.jpg
```

In order to use the web api version, follow these steps:

```bash
> ./main -u objectDetection -m 1
```

The web api will run on port 8000 by default.