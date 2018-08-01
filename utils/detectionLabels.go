package utils

import (
	"os"
	"log"
	"bufio"
	"strings"
	)

func (l *Labels) Load(uc string) {
	_, labelFileTemp := GenerateUseCasePath(uc, "label")
	file, err := os.Open(labelFileTemp)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		l.Labels = append(l.Labels, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Printf("ERROR: failed to read %s: %v", l.LabelsDir, err)
	}
}

func (l *Labels) GetLabel(idx int, probabilities []float32, classes []float32) (string, float32) {
	index := int(classes[idx])
	s := strings.Split(l.Labels[index], ":")[1]
	label := s[2 : len(s)-1]
	return strings.Title(label), probabilities[idx]*100.0
}

