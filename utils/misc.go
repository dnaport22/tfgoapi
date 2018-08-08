package utils

import (
		"io/ioutil"
	"log"
		"strings"
	"fmt"
)

func AvailableUseCases()[]string {
	var usecases []string
	fl, err := ioutil.ReadDir("data/useCases")
	if err != nil {
		log.Fatal(err)
	}
	for _, f := range fl {
		usecases = append(usecases, f.Name())
	}
	return usecases
}

func IsValidUseCase(cases []string, c string)bool {
	return contains(cases, c)
}

func GenerateUseCasePath(uc string, data string) (string, string) {
	baseLoc := "data/useCases"
	var ext string
	if data == "model" {
		ext = ".pb"
	}
	if data == "label" {
		ext = ".pbtxt"
	}
	fileTemp := fmt.Sprintf("%s/%s/%s%s%s",
		baseLoc, uc, uc, strings.Title(data), ext)

	return baseLoc, fileTemp
}


// helper function to check if
// elem x exists in the given []string
func contains(s []string, e string) bool {
	for _, a := range s {
		if a == e {
			return true
		}
	}
	return false
}
