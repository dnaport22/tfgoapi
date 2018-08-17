package core

import ("log"
	u "tfGraphApi/utils"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/gorilla/mux"
	"net/http"
	"io/ioutil"
	"image"
	"bytes"
	"path/filepath"
	"fmt"
			)


func RunAzureModel(im u.Img) (int, int){

	input := AzureGraph.Graph.Operation("Placeholder")
	out := AzureGraph.Graph.Operation("loss")

	output, err := AzureGraph.Session.Run(
		map[tf.Output]*tf.Tensor{
			input.Output(0): im.NormalisedImgTensor()[0],
		},
		[]tf.Output{
			out.Output(0),
		},
		nil)
	if err != nil {
		log.Fatal(err)
	}
	probabilities := output[0].Value().([][]float32)[0]

	return int(probabilities[0]*100), int(probabilities[1]*100)
}

func RunApi(port string) {
	//Initialising routes
	router := mux.NewRouter()
	router.HandleFunc("/get-people", GetPeople).Methods("POST")
	log.Fatal(http.ListenAndServe(port, router))
}

func RunLocal(dir string) {
	// Reading local file into byte slices
	fl, _ := ioutil.ReadDir(dir)
	for i := 0; i < len(fl); i++ {
		if u.AvailableFormat(filepath.Ext(fl[i].Name())) {
			log.Print("Processing: " + dir + "/" + fl[i].Name())
			// Resize and crop the srcImage to fill the 100x100px area.
			b, _ := ioutil.ReadFile(dir + "/" + fl[i].Name())
			// Decoding bytes into image object
			img, _, _ := image.Decode(bytes.NewReader(b))
			// Setting image bytes for processing
			im.ImageBytes = b
			// Setting image object for processing
			im.ImgObject = img
			// Initialising image tensor
			im.SetImgTensor()
			// Executing detection job
			runTfSession(i, fl[i].Name())
		}
	}
	peopleCount := 0
	for _, v := range Trackers {
		fmt.Println(v)
		peopleCount += len(v.Centroids)
	}
	fmt.Printf("Total people count: %v", peopleCount)
}

func runTfSession(frameId int, frameName string) {
	// Run tensorflow session
	output, err := TfGraph.Session.Run(
		map[tf.Output]*tf.Tensor{
			TfGraph.Graph.Operation("image_tensor").Output(0): im.ImgTensor,
		},
		[]tf.Output{
			TfGraph.Graph.Operation("detection_scores").Output(0),
			TfGraph.Graph.Operation("detection_classes").Output(0),
			TfGraph.Graph.Operation("detection_boxes").Output(0),
			TfGraph.Graph.Operation("num_detections").Output(0),
		},
		nil)
	if err != nil {
		log.Fatal(err)
	}

	// Outputs
	probabilities := output[0].Value().([][]float32)[0]
	classes := output[1].Value().([][]float32)[0]
	boxes := output[2].Value().([][][]float32)[0]

	// Transform the decoded YCbCr JPG image into RGBA
	b := im.ImgObject.Bounds()
	img := image.NewRGBA(b)

	curObj := 0
	var boundingBox [][]float32
	for probabilities[curObj] > 0.8 && classes[curObj] == 1 {
		x1 := float32(img.Bounds().Max.X) * boxes[curObj][1]
		x2 := float32(img.Bounds().Max.X) * boxes[curObj][3]
		y1 := float32(img.Bounds().Max.Y) * boxes[curObj][0]
		y2 := float32(img.Bounds().Max.Y) * boxes[curObj][2]
		centroidX, centroidY := u.CalculateCentroid(x1, y1, x2, y2)
		boundingBox = append(boundingBox, []float32{x1, y1, x2 ,y2, centroidX, centroidY})
		// Detection boundary Y <- 150-200
		for i, _ := range PreviousBox {
			for j, _ := range boundingBox {
				distFromPrev := u.EuclideanDistance(
					PreviousBox[i][4], boundingBox[j][4],
					PreviousBox[i][5], boundingBox[j][5])
				fmt.Println(distFromPrev)
				if distFromPrev > 5 {
					updateTracker(frameName, boundingBox, distFromPrev)
				}
			}
		}
		curObj++
		PreviousBox = boundingBox
	}
}

func updateTracker(fnm string, bbox [][]float32, dist float32) {
	Trackers = append(Trackers, u.TrackableObject{
		ObjectId: u.GenUuid(),
		FrameName: fnm,
		Counted: true,
		Centroids: bbox,
		DistanceFromPrevious: dist,
	})
}