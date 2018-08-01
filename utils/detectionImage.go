package utils

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	_ "image/jpeg"
	_ "image/png"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"log"
	)

func decodeJpegGraph() (graph *tf.Graph, input, output tf.Output, err error) {
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output = op.ExpandDims(s,
		op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)),
		op.Const(s.SubScope("make_batch"), int32(0)))
	graph, err = s.Finalize()
	return graph, input, output, err
}

func decodePngGraph() (graph *tf.Graph, input, output tf.Output, err error) {
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output = op.ExpandDims(s,
		op.DecodePng(s, input, op.DecodePngChannels(3)),
		op.Const(s.SubScope("make_batch"), int32(0)))
	graph, err = s.Finalize()
	return graph, input, output, err
}

func (i *Img) SetImgLoc(l string) {
	i.ImgLoc = l
}

func (i *Img) GetImgLoc()string {
	return i.ImgLoc
}

func (i *Img) SetImgType(t string) {
	i.ImgType = t
}

func (i *Img) GetImgType()string {
	return i.ImgType
}

func (i *Img) SetImgTensor() {
	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(i.ImageBytes))
	if err != nil {
		log.Fatal(err)
	}
	// Creates a tensorflow graph to decode the png Img
	graph, input, output, err := decodePngGraph()
	if err != nil {
		log.Fatal(err)
	}
	// Execute that graph to decode this one Img
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		log.Fatal(err)
	}
	i.ImgTensor = normalized[0]
}

func (i *Img) GetImageTensor()*tf.Tensor  {
	return i.ImgTensor
}
