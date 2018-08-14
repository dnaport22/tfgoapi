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
	"image/jpeg"
	"github.com/disintegration/imaging"
		azure "tfGraphApi/third-party/azurevision"
	"path/filepath"
)

var im u.Img
var model u.Model
var labels u.Labels
var detectionGraph u.DetectionGraph

var azureTfModel u.Model
var azureTfLabels u.Labels
var azureGraph u.DetectionGraph

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
	// Empty byte buffer
	data := new(bytes.Buffer)
	// Decoding byte slice into image.Image
	img, _, _ := image.Decode(r.Body)
	// Resizing image to lower feature size
	img = imaging.Resize(img, 227, 227, imaging.Lanczos)

	// Encoding image into jpeg and loading it on empty byte buffer
	jpeg.Encode(data, img, nil)
	// Setting image bytes for processing
	im.ImageBytes = data.Bytes()
	// Setting image object for processing
	im.ImgObject = img
	// Initialising image tensor
	im.SetImgTensor()

	detection := runTfSession()

	w.WriteHeader(http.StatusOK)
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(detection); err != nil {
		panic(err)
	}
}

func runAzureModel() (int, int){

	input := azureGraph.Graph.Operation("Placeholder")
	out := azureGraph.Graph.Operation("loss")

	output, err := azureGraph.Session.Run(
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

func runTfSession() []u.DetectedObject {
	// Run tensorflow session
	output, err := detectionGraph.Session.Run(
		map[tf.Output]*tf.Tensor{
			detectionGraph.ImageOperation.Output(0): im.ImgTensor,
		},
		[]tf.Output{
			detectionGraph.DetectionScore.Output(0),
			detectionGraph.DetectionClasses.Output(0),
			detectionGraph.BoundingBoxes.Output(0),
			detectionGraph.NumDetections.Output(0),
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

	// Initialising Azure Api
	vision, _ := azure.New("7b11a4bfca114ca2949c1d0f659e3aed",
		"https://uksouth.api.cognitive.microsoft.com/vision/v1.0")

	var detectedObject []u.DetectedObject

	curObj := 0
	for probabilities[curObj] > 0.4 {
		x1 := float32(img.Bounds().Max.X) * boxes[curObj][1]
		x2 := float32(img.Bounds().Max.X) * boxes[curObj][3]
		y1 := float32(img.Bounds().Max.Y) * boxes[curObj][0]
		y2 := float32(img.Bounds().Max.Y) * boxes[curObj][2]

		// cropping image for azure vision api
		cropped := imaging.Crop(im.ImgObject, image.Rectangle{
			image.Point{int(x1), int(y1)},
			image.Point{int(x2), int(y2)}})

		// Empty buffer to store azure vision api results
		azureData := new(bytes.Buffer)
		// Encoding cropped image into jpeg
		jpeg.Encode(azureData, cropped, nil)
		// Calling azure vision api
		azCloudVisionOut := make(chan azure.VisionResult)
		azCustomVisionOut := make(chan []int)

		var face azure.Face
		var indianClothes int
		var westernClothes int

		go func(data []byte) {
			out, _ := vision.AnalyzeImage(data, azure.VisualFeatures{Faces:true})
			azCloudVisionOut <- out
		}(azureData.Bytes())

		go func() {
			iC, wC := runAzureModel()
			azCustomVisionOut <- []int{iC, wC}
		}()

		for i := 0; i < 2; i++ {
			select {
			case msg1 := <-azCloudVisionOut:
				if len(msg1.Faces) > 0 {
					face = msg1.Faces[0]
				}
			case msg2 := <-azCustomVisionOut:
				westernClothes = msg2[1]
				indianClothes = msg2[0]
			}
		}
		// Making variable short to read
		faceBox := face.FaceRectangle

		label, prob := labels.GetLabel(curObj, probabilities, classes)
		detectedObject = append(detectedObject, u.DetectedObject{
			ObjectId: curObj, Label: label, Probability: int(prob),
			Age: face.Age, Gender: face.Gender,
			ObjectBox: &u.BBox{MinX: x1, MinY: y1, MaxX: x2, MaxY: y2},
			FaceBox: &u.BBox{MinX: float32(faceBox.Left), MinY: float32(faceBox.Top),
			MaxX: float32(faceBox.Width),
			MaxY: float32(faceBox.Height)},
			Clothing: &u.AzureClothing{Indian: indianClothes, Western: westernClothes},
			NumberOfPeopleDetected: u.GetObjectLen(probabilities),
		})
		curObj++
	}

	return detectedObject
}

func runLocal(dir string) {
	// Reading local file into byte slices
	fl, _ := ioutil.ReadDir(dir)
	for i := 0; i < len(fl); i++ {
		if u.AvailableFormat(filepath.Ext(fl[i].Name())) {
			log.Print("Processing: " + dir + "/" + fl[i].Name())
			// Resize and crop the srcImage to fill the 100x100px area.
			b, _ := ioutil.ReadFile(dir + "/" + fl[i].Name())
			// Empty byte buffer
			data := new(bytes.Buffer)
			// Decoding bytes into image object
			img, _, _ := image.Decode(bytes.NewReader(b))
			// Resizing image to lower feature size
			img = imaging.Resize(img, 227, 227, imaging.Lanczos)
			// Encoding image into jpeg and loading it on empty byte buffer
			jpeg.Encode(data, img, nil)
			// Setting image bytes for processing
			im.ImageBytes = data.Bytes()
			// Setting image object for processing
			im.ImgObject = img
			// Initialising image tensor
			im.SetImgTensor()
			// Executing detection job
			detection := runTfSession()
			// Printing results
			fmt.Print(detection, "\n")
		}
	}
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
	imgDir := flag.String("img-dir", "", "Path of a JPG image to use for input")
	azureVision := flag.String("az", "", "Use Azure Vision Model")
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

	if len(*azureVision) > 0 {
		azureTfModel.Load(*azureVision)
		azureTfLabels.Load(*azureVision)

		// Construct an in-memory graph from the serialized form.
		azureGraph.Graph = tf.NewGraph()
		if err := azureGraph.Graph.Import(azureTfModel.Model, ""); err != nil {
			log.Fatal(err)
		}

		// Create a session for inference over graph.
		session, err := tf.NewSession(azureGraph.Graph, nil)
		azureGraph.Session = session
		if err != nil {
			log.Fatal(err)
		}
		defer session.Close()
	}

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
			runLocal(*imgDir)
		}
	}
}