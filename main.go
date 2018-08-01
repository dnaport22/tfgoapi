package main

import (
		u "tfGraphApi/utils"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"log"
	"fmt"
	_ "image/jpeg"
	_ "image/png"
	"net/http"
		"io/ioutil"
					"bytes"
	"image"
	"encoding/json"
	"github.com/gorilla/mux"
				"flag"
	"github.com/disintegration/imaging"
		"image/jpeg"
		azure "tfGraphApi/thirdparty/azurevision"
)


var im u.Img
var model u.Model
var labels u.Labels
var detectionGraph u.DetectionGraph

func loadGraph() {
	graph := detectionGraph.Graph
	detectionGraph.ImageOperation = graph.Operation("image_tensor")
	detectionGraph.DetectionScore = graph.Operation("detection_scores")
	detectionGraph.DetectionClasses = graph.Operation("detection_classes")
	detectionGraph.BoundingBoxes = graph.Operation("detection_boxes")
	detectionGraph.NumDetections = graph.Operation("num_detections")
	log.Print(fmt.Sprintf("Loaded graph\n\tName: %s\n\tDescription: %s", model.Name, model.Description))
}


func GetPeople(w http.ResponseWriter, r *http.Request) {
	// reading image data into byte slices
	contents, _ := ioutil.ReadAll(r.Body)
	// Augmenting contents
	r.Body = ioutil.NopCloser(bytes.NewReader(contents))
	defer r.Body.Close()

	// Decoding byte slice into image.Image
	img, _, _ := image.Decode(r.Body)

	im.ImageBytes = contents
	im.ImgObject = img
	im.SetImgTensor()
	detection := runTfSession()

	w.WriteHeader(http.StatusOK)
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(detection); err != nil {
		panic(err)
	}
}


func runTfSession() []u.DetectedObject {
	session := detectionGraph.Session
	imageInput := detectionGraph.ImageOperation
	detectionScore := detectionGraph.DetectionScore
	detectionClasses := detectionGraph.DetectionClasses
	boundingBoxes := detectionGraph.BoundingBoxes
	numDetection := detectionGraph.NumDetections

	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			imageInput.Output(0): im.ImgTensor,
		},
		[]tf.Output{
			detectionScore.Output(0),
			detectionClasses.Output(0),
			boundingBoxes.Output(0),
			numDetection.Output(0),
		},
		nil)
	if err != nil {
		log.Fatal(err)
	}

	// Outputs
	probabilities := output[0].Value().([][]float32)[0]
	classes := output[1].Value().([][]float32)[0]
	boxes := output[2].Value().([][][]float32)[0]

	curObj := 0
	i := im.ImgObject
	// Transform the decoded YCbCr JPG image into RGBA
	b := i.Bounds()
	img := image.NewRGBA(b)
	vision, _ := azure.New("<KEY>",
		"https://uksouth.api.cognitive.microsoft.com/vision/v1.0")

	var detectedObject []u.DetectedObject

	for probabilities[curObj] > 0.4 {
		x1 := float32(img.Bounds().Max.X) * boxes[curObj][1]
		x2 := float32(img.Bounds().Max.X) * boxes[curObj][3]
		y1 := float32(img.Bounds().Max.Y) * boxes[curObj][0]
		y2 := float32(img.Bounds().Max.Y) * boxes[curObj][2]

		cropped := imaging.Crop(i, image.Rectangle{
			image.Point{int(x1), int(y1)},
			image.Point{int(x2), int(y2)}})

		azureData := new(bytes.Buffer)
		jpeg.Encode(azureData, cropped, nil)
		azOut, _ := vision.AnalyzeImage(azureData.Bytes(), azure.VisualFeatures{Faces:true})

		label, prob := labels.GetLabel(curObj, probabilities, classes)
		var face azure.Face
		if len(azOut.Faces) > 0 {
			face = azOut.Faces[0]
		}
		faceBox := face.FaceRectangle
		detectedObject = append(detectedObject, u.DetectedObject{
			ObjectId: curObj, Label: label, Probability: int(prob),
			Age: face.Age, Gender: face.Gender,
			ObjectBox: &u.BBox{MinX: int(x1), MinY: int(y1), MaxX: int(x2), MaxY: int(y2)},
			FaceBox: &u.BBox{MinX: faceBox.Left, MinY: faceBox.Top, MaxX: faceBox.Width, MaxY: faceBox.Height},
		})
		curObj++
	}

	return detectedObject
}


func runLocal(fnm string) {
	// Reading local file into byte slices
	b, _ := ioutil.ReadFile(fnm)
	im.ImageBytes = b
	img, _, _ := image.Decode(bytes.NewReader(b))
	im.ImgObject = img
	im.SetImgTensor()
	detection := runTfSession()
	fmt.Print(detection)
}


func runApi(port string) {
	//Initialising routes
	router := mux.NewRouter()
	router.HandleFunc("/get-people", GetPeople).Methods("POST")
	log.Fatal(http.ListenAndServe(port, router))
}


func main() {
	mode := flag.Int("m", 0, "Run time mode: 0 => local, 1 => api")
	useCase := flag.String("u", "", "Use case to run")
	imgFile := flag.String("img", "", "Path of a JPG image to use for input")
	//useAzureVision := flag.Bool("az", true, "Use Azure Vision: false => no, true => yes")
	flag.Parse()

	modelToRun := *useCase
	availableUseCases := u.AvailableUseCases()

	if !u.IsValidUseCase(availableUseCases, modelToRun) {
		log.Fatal(
			fmt.Sprintf("Invalid use case.\n\tAvailable use cases: %v",
				availableUseCases))
	}

	model.Load(modelToRun)
	labels.Load(modelToRun)

	// Construct an in-memory graph from the serialized form.
	detectionGraph.Graph = tf.NewGraph()
	if err := detectionGraph.Graph.Import(model.Model, ""); err != nil {
		log.Fatal(err)
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(detectionGraph.Graph, nil)
	detectionGraph.Session = session
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	loadGraph()

	if *mode > 1 {
		log.Fatal("Mode not available.\n\tAvailable modes: 0 => local, 1 => api")
	} else {
		if *mode == 1 {
			log.Print("Running web api")
			runApi(":8000")
		} else {
			log.Print("Running bash api")
			runLocal(*imgFile)
		}
	}
}