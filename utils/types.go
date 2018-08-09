package utils

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"image"
)

type DetectedObject struct {
	ObjectId int `json:"Id,omitempty"`
	Label string `json:"Label,omitempty"`
	Probability int `json:"Probability,omitempty"`
	ObjectBox *BBox `json:"ObjectBoundingBox,omitempty"`
	Age int `json:"Age,omitempty"`
	Gender string `json:"Gender,omitempty"`
	FaceBox *BBox `json:"FaceBoundingBox,omitempty"`
	Clothing *AzureClothing `json:"Clothing,omitempty"`
	NumberOfPeopleDetected int `json:"NumberOfPeopleDetected,omitempty"`
}

type DetectionGraph struct {
	Graph *tf.Graph
	Session *tf.Session
	ImageOperation *tf.Operation
	DetectionScore *tf.Operation
	DetectionClasses *tf.Operation
	BoundingBoxes *tf.Operation
	NumDetections *tf.Operation
}

type Img struct {
	ImgLoc string
	ImgType string
	ImgName string
	ImgTensor *tf.Tensor
	ImgObject image.Image
	ImageBytes []byte
}

type Labels struct {
	Labels[] string
	LabelsDir string
}

type Model struct {
	Description string
	Path string
	UseCase string
	Model []byte
	Name string
	Base string
}

type BBox struct {
	MinX int `json:"x1,omitempty"`
	MinY int `json:"y1,omitempty"`
	MaxX int `json:"x2,omitempty"`
	MaxY int `json:"y2,omitempty"`
}

type AzureClothing struct {
	Indian int `json:"Indian"`
	Western int `json:"Western"`
}
