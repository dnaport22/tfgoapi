package utils

import (
	"log"
	"io/ioutil"
	"fmt"
	"os"
	"encoding/json"
)

func (m *Model) Load(p string) {
	baseLoc, modelFileTemp := GenerateUseCasePath(p, "model")
	model, err := ioutil.ReadFile(modelFileTemp)
	if err != nil {
		log.Fatal(err)
	}
	m.Path = modelFileTemp
	m.Base = baseLoc
	m.UseCase = p
	m.Model = model
	m.LoadManifest()
}

func (m *Model) LoadManifest() {
	manifestFileTemp := fmt.Sprintf("%s/%s/manifest.json", m.Base, m.UseCase)
	manifest, err := os.Open(manifestFileTemp)
	if err != nil {
		log.Fatal(err)
	}
	defer manifest.Close()
	byteValue, _ := ioutil.ReadAll(manifest)
	var manifestData struct{
		Description string `json:"description"`
		Name string `json:"model_name"`
	}
	json.Unmarshal(byteValue, &manifestData)
	m.Description  = manifestData.Description
	m.Name  = manifestData.Name
}