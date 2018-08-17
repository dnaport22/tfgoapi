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
	"math"
	azure "tfGraphApi/third-party/azurevision"
	"os"
	"encoding/csv"
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
	// Initialising Azure Api
	vision, _ := azure.New("7b11a4bfca114ca2949c1d0f659e3aed",
		"https://uksouth.api.cognitive.microsoft.com/vision/v1.0")
	var output [][]string
	for _, v := range Trackers {
		peopleCount += len(v.Centroids)
		azCloudVisionOut := make(chan azure.VisionResult)
		azCustomVisionOut := make(chan []int)
		var face azure.Face
		var clothing u.Clothing
		go func(data []byte) {
			log.Println("Running age-gender detection job")
			out, _ := vision.AnalyzeImage(data, azure.VisualFeatures{Faces:true, Description:true})
			azCloudVisionOut <- out
		}(v.ImageData)
		go func(img u.Img) {
			log.Println("Running clothing classification job")
			iC, wC := RunAzureModel(img)
			azCustomVisionOut <- []int{iC, wC}
		}(im)
		for i := 0; i < 2; i++ {
			select {
			case msg1 := <-azCloudVisionOut:
				if len(msg1.Faces) > 0 {
					face = msg1.Faces[0]
				}
			case msg2 := <-azCustomVisionOut:
				clothing = u.Clothing{
					&u.AzureClothing{Indian: int(msg2[1]), Western: int(msg2[0])},
					nil}
			}
		}
		output = append(output, []string{
			v.FrameName,
			face.Gender,
			u.AgeGroup(face.Age),
			clothing.WhichClothing(),
		})
	}
	dumpToCsv(output, dir)
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
		if len(Trackers) == 0 {
			updateTracker(frameName, boundingBox, 0, im.ImageBytes)
		} else {
			for i, _ := range PreviousBox {
				for j, _ := range boundingBox {
					distFromPrev := u.EuclideanDistance(
						PreviousBox[i][4], boundingBox[j][4],
						PreviousBox[i][5], boundingBox[j][5])
					if math.Round(float64(distFromPrev)) > float64(5) {
						updateTracker(frameName, boundingBox, distFromPrev, im.ImageBytes)
					}
				}
			}
		}

		curObj++
		PreviousBox = boundingBox
	}
}

func updateTracker(fnm string, bbox [][]float32, dist float32, img []byte) {
	Trackers = append(Trackers, u.TrackableObject{
		ObjectId: u.GenUuid(),
		FrameName: fnm,
		Counted: true,
		Centroids: bbox,
		DistanceFromPrevious: dist,
		ImageData: img,
	})
}

func dumpToCsv(output [][]string, dir string)  {
	// Writing to CSV
	log.Println("Writing results to csv")
	file, _ := os.Create(fmt.Sprintf("%s_results.csv", dir))
	writer := csv.NewWriter(file)
	defer writer.Flush()
	// Headers
	headers := []string{"Frame Name" , "Gender", "Age", "Clothing"}
	writer.Write(headers)
	for _, v := range output {
		writer.Write(v)
	}
	log.Println(fmt.Sprintf("Finished writing results to csv -> %s_results.csv", dir))
}