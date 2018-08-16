package core

import ("log"
u "tfGraphApi/utils"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/gorilla/mux"
	"net/http"
)

var AzureTfModel u.Model
var AzureTfLabels u.Labels
var AzureGraph u.DetectionGraph

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