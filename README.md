# TFGraphApi
TensorFlow Graph Api to execute tensflow based pre-trained models. 
This project already has 2 models, i.e. Face Detection and Object Detection.
This also a TF-Slim model trained using Azure Custom Vision for cloth classification.

This project is a part of Deep Fashion Project and is intended to work along with all other smart micro-services under the project.

# How to run
## Prerequisites
 - [Tensorflow](https://www.tensorflow.org/install/install_c)
 - [Golang](https://golang.org/)

Once the all of the requirements are satisfied, we can start playing with the api. In order to get the codebase and execute pre-complied binary, follow the steps:

```bash
# install required dependencies
> ./install
# getting the codebase
> git clone git@scm.acrotrend.com:retail-in-store-shopper-insights/tf-graph-api.git tfGraphApi
> cd tfGraphApi
# running api
> go run -u objectDetection \ #selecting the base use case
 -az clothDetection \ #selecting tf-slim model trained on azure 
 -img-dir IMAGES \ #input image dir \
  -c #remove image from dir after processing
```