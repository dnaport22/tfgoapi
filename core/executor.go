package core

import ("log"
	u "../utils"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/gorilla/mux"
	"net/http"
	"io/ioutil"
	"image"
			"fmt"
			"os"
		azure "../third-party/azurevision"
	"math"
	"time"
	"path/filepath"
	"bytes"
	"strings"
	"strconv"
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
	if len(fl) > 0 {
		log.Println("Reading image directory" + dir)
		for i := 0; i < len(fl); i++ {
			if u.AvailableFormat(filepath.Ext(fl[i].Name())) {
				log.Println("Processing: " + dir + "/" + fl[i].Name())
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
				if CleaningDir {
					os.Remove(dir + "/" + fl[i].Name())
				}
			}
		}
		output := additionalDetection()
		fmt.Println(output)
		dumpToCsv(output, dir)
		RunLocal(dir)
	} else {
		log.Println("Empty image directory " + dir)
		ImgDirReadAttempts++
		log.Println("Waiting for image directory " + dir)
		time.Sleep(5000 * time.Millisecond)
		RunLocal(dir)
	}
}

func additionalDetection()[][]string {
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
		go func(data u.TrackableObject) {
			log.Printf("Running age-gender detection job %s", data.FrameName)
			out, _ := vision.AnalyzeImage(data.ImageData, azure.VisualFeatures{Faces:true, Description:true})
			azCloudVisionOut <- out
		}(v)
		go func(img u.Img) {
			log.Printf("Running clothing classification job %s", img.ImgName)
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
			strconv.Itoa(clothing.Group.Indian),
			strconv.Itoa(clothing.Group.Western),
		})
	}
	return output
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
	for probabilities[curObj] > float32(ProbabilityThreshold) && classes[curObj] == 1 {
		x1 := float32(img.Bounds().Max.X) * boxes[curObj][1]
		x2 := float32(img.Bounds().Max.X) * boxes[curObj][3]
		y1 := float32(img.Bounds().Max.Y) * boxes[curObj][0]
		y2 := float32(img.Bounds().Max.Y) * boxes[curObj][2]
		centroidX, centroidY := u.CalculateCentroid(x1, y1, x2, y2)
		boundingBox = append(boundingBox, []float32{x1, y1, x2 ,y2, centroidX, centroidY})
		updateTracker(frameName, boundingBox, im.ImageBytes)
		if len(Trackers) == 0 {
			updateTracker(frameName, boundingBox, im.ImageBytes)
		} else {
			for i, _ := range PreviousBox {
				for j, _ := range boundingBox {
					distFromPrev := u.EuclideanDistance(
						PreviousBox[i][4], boundingBox[j][4],
						PreviousBox[i][5], boundingBox[j][5])
					if math.Round(float64(distFromPrev)) > float64(50) {
						updateTracker(frameName, boundingBox, im.ImageBytes)
					}
				}
			}
		}

		curObj++
		PreviousBox = boundingBox
	}
}

func updateTracker(fnm string, bbox [][]float32, img []byte) {
	Trackers = append(Trackers, u.TrackableObject{
		ObjectId: u.GenUuid(),
		FrameName: fnm,
		Counted: true,
		Centroids: bbox,
		ImageData: img,
	})
}

func dumpToCsv(output [][]string, dir string)  {
	// Writing to CSV
	log.Println("Writing results to csv")
	file, _ := os.OpenFile(fmt.Sprintf("%s_results.csv", dir), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	for _, v := range output {
		file.WriteString(strings.Join(v, ",") + "\n")
	}
	log.Println(fmt.Sprintf("Finished writing results to csv -> %s_results.csv", dir))
}