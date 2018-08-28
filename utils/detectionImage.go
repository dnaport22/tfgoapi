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

func decodeBmpGraph() (graph *tf.Graph, input, output tf.Output, err error) {
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output = op.ExpandDims(s,
		op.DecodeBmp(s, input, op.DecodeBmpChannels(int64(3))),
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

func ConstructGraphToNormaliseImage() (graph *tf.Graph, input, output tf.Output, err error) {
	// Some constants specific to the pre-trained model at:
	// https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
	//
	// - The model was trained after with images scaled to 224x224 pixels.
	// - The colors, represented as R, G, B in 1-byte each were converted to
	//   float using (value - Mean)/Scale.
	const (
		H, W  = 224, 224
		Mean  = float32(117)
		Scale = float32(1)
	)
	// - input is a String-Tensor, where the string the JPEG-encoded image.
	// - The inception model takes a 4D tensor of shape
	//   [BatchSize, Height, Width, Colors=3], where each pixel is
	//   represented as a triplet of floats
	// - Apply normalization on each pixel and use ExpandDims to make
	//   this single image be a "batch" of size 1 for ResizeBilinear.
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output = op.Div(s,
		op.Sub(s,
			op.ResizeBilinear(s,
				op.ExpandDims(s,
					op.Cast(s,
						op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)), tf.Float),
					op.Const(s.SubScope("make_batch"), int32(0))),
				op.Const(s.SubScope("size"), []int32{H, W})),
			op.Const(s.SubScope("mean"), Mean)),
		op.Const(s.SubScope("scale"), Scale))
	graph, err = s.Finalize()
	return graph, input, output, err
}

func (i *Img) NormalisedImgTensor()[]*tf.Tensor {
	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(i.ImageBytes))
	if err != nil {
		log.Fatal(err)
	}
	// Creates a tensorflow graph to decode the png Img
	graph, input, output, err := ConstructGraphToNormaliseImage()
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
	return normalized
}

func (i *Img) SetImgTensor() {
	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(i.ImageBytes))
	if err != nil {
		log.Fatal(err)
	}
	// Creates a tensorflow graph to decode the png Img
	graph, input, output, err := decodeJpegGraph()
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
