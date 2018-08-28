package utils

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"image"
	)

type Clothing struct {
	Group *AzureClothing `json:"ClothingGroup,omitempty"`
	Assoc []string
}

type Gender struct {
	Label string `json:"Gender,omitempty"`
	Assoc []string
}

type DetectedObject struct {
	ObjectId int `json:"Id,omitempty"`
	Label string `json:"Label,omitempty"`
	Probability int `json:"Probability,omitempty"`
	ObjectBox *BBox `json:"ObjectBoundingBox,omitempty"`
	Age int `json:"Age,omitempty"`
	Gender *Gender `json:"Gender,omitempty"`
	FaceBox *BBox `json:"FaceBoundingBox,omitempty"`
	Clothing *Clothing `json:"Clothing,omitempty"`
	NumberOfPeopleDetected int `json:"NumberOfPeopleDetected,omitempty"`
}

type TrackableObject struct {
	ObjectId string `json:"Id,omitempty"`
	FrameName string `json:"frame_name,omitempty"`
	Counted bool `json:"Counted,omitempty"`
	Centroids [][]float32
	DistanceFromPrevious float32
	ImageData []byte
}

type Train struct {
	Tx float32
	Ty float32
	FrameName string
}


type Neighbors struct {
	Nx float32
	Ny float32
	Dist float32
	FrameName string
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
	MinX float32 `json:"x1,omitempty"`
	MinY float32 `json:"y1,omitempty"`
	MaxX float32 `json:"x2,omitempty"`
	MaxY float32 `json:"y2,omitempty"`
}

type AzureClothing struct {
	Indian int `json:"Indian"`
	Western int `json:"Western"`
}
