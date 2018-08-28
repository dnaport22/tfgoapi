package utils

import (
		"io/ioutil"
	"log"
		"strings"
	"fmt"
	"os/exec"
	"math"
)

var fileFormats  = []string{".jpg", ".jpeg", ".png"}

func AvailableUseCases()[]string {
	var usecases []string
	fl, err := ioutil.ReadDir("tfGraphApi/data/useCases")
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
	baseLoc := "tfGraphApi/data/useCases"
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

func GetObjectLen(object []float32)int {
	curObj := 0
	for object[curObj] > 0.4 {
		curObj++
	}
	return curObj
}

func GetNumPeopleDetected(probabilities []float32, classes []float32)int {
	curObj := 0
	people := 0
	for probabilities[curObj] > 0.5 && classes[curObj] == 1 {
		curObj++
		people++
	}
	return people
}

func AvailableFormat(fmt string)bool {
	return contains(fileFormats, fmt)
}

func SanitiseString(str string)string {
	if len(str) < 1 {
		return "null"
	}
	return str
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

func CalculateCentroid(x1 float32, y1 float32, x2 float32, y2 float32)(float32, float32) {
	return (x1 + x2) / 2, (y1 + y2) / 2
}

func EuclideanDistance(x1 float32, y1 float32, x2 float32, y2 float32)float32 {
	dist := math.Sqrt(
		math.Pow((float64(x1)-float64(x2)), 2) + math.Pow((float64(y1)-float64(y2)), 2))
	return float32(dist) / 100 *2
}

func CDist(coords_a [][]float32, coords_b []float32)[]float32 {
	var dist []float32
	for _, v := range coords_a {
		d := math.Sqrt(
			math.Pow((float64(v[4])-float64(coords_b[4])), 2) +
				math.Pow((float64(v[5])-float64(coords_b[5])), 2))
		dist = append(dist, float32(d)/100*2)
	}

	return dist
}

func GenUuid()string {
	out, err := exec.Command("uuidgen").Output()
	if err != nil {
		log.Fatal(err)
	}
	return string(out)
}

func GetNeighbors(trainingInstance []float32, trainingSet []Train)[]Neighbors {
	var neighbors []Neighbors
	var Knn []Neighbors
	for i, _ := range trainingSet {
		dist := EuclideanDistance(
			trainingInstance[0], trainingSet[i].Tx,
			trainingInstance[1], trainingSet[i].Ty)
		if dist > 5 {
			neighbors = append(neighbors, Neighbors{
				Nx:trainingSet[i].Tx, Ny:trainingSet[i].Ty,
				FrameName:trainingSet[i].FrameName,
				Dist:dist,
			})
		}

	}
	for i := 0; i < 3; i++ {
		Knn = append(Knn, neighbors[i])
	}
	return neighbors
}
