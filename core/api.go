package core

import (
	"net/http"
	"io/ioutil"
	"bytes"
	"image"
	"github.com/disintegration/imaging"
	"image/jpeg"
			)

func GetPeople(w http.ResponseWriter, r *http.Request) {
	// reading image data into byte slices
	contents, _ := ioutil.ReadAll(r.Body)
	// Augmenting contents
	r.Body = ioutil.NopCloser(bytes.NewReader(contents))
	defer r.Body.Close()
	// Empty byte buffer
	data := new(bytes.Buffer)
	// Decoding byte slice into image.Image
	img, name, _ := image.Decode(r.Body)
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

	runTfSession(0, name)

	//w.WriteHeader(http.StatusOK)
	//w.Header().Set("Content-Type", "application/json")
	//if err := json.NewEncoder(w).Encode(detection); err != nil {
	//	panic(err)
	//}
}
