package main

import (
	u "./utils"
	c "./core"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"log"
	"fmt"
	_ "image/jpeg"
	_ "image/png"
	"flag"
	)

var model u.Model
var labels u.Labels

func main() {
	mode := flag.Int("m", 0, "Run time mode: 0 => local, 1 => api")
	useCase := flag.String("u", "", "Use case to run")
	imgDir := flag.String("img-dir", "", "Path of a JPG image to use for input")
	azureVision := flag.String("az", "", "Use Azure Vision Model")
	pThresh := flag.Float64("p", 0.6, "Probability threshold")
	clean := flag.Bool("c", true, "Clean after reading: 0 => yes, 1 => no")
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
		c.AzureTfModel.Load(*azureVision)
		c.AzureTfLabels.Load(*azureVision)

		// Construct an in-memory graph from the serialized form.
		c.AzureGraph.Graph = tf.NewGraph()
		if err := c.AzureGraph.Graph.Import(c.AzureTfModel.Model, ""); err != nil {
			log.Fatal(err)
		}

		// Create a session for inference over graph.
		session, err := tf.NewSession(c.AzureGraph.Graph, nil)
		c.AzureGraph.Session = session
		if err != nil {
			log.Fatal(err)
		}
		defer session.Close()
	}

	// Construct an in-memory graph from the serialized form.
	c.TfGraph.Graph = tf.NewGraph()
	if err := c.TfGraph.Graph.Import(model.Model, ""); err != nil {
		log.Fatal(err)
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(c.TfGraph.Graph, nil)
	c.TfGraph.Session = session
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	if *mode > 1 {
		log.Fatal("Mode not available.\n\tAvailable modes: 0 => local, 1 => api")
	} else {
		if *mode == 1 {
			log.Print("Running web api")
			c.RunApi(":8000")
		} else {
			log.Print("Running bash api")
			c.ProbabilityThreshold = *pThresh
			c.CleaningDir = *clean
			c.RunLocal(*imgDir)
		}
	}
}