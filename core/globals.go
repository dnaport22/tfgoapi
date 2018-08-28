package core

import (u "../utils")

var im u.Img
var AzureTfModel u.Model
var AzureTfLabels u.Labels
var AzureGraph u.DetectionGraph
var TfGraph u.DetectionGraph
var Trackers []u.TrackableObject
var ObjectsTracked int
var PreviousBox [][]float32
var ProbabilityThreshold float64
var CleaningDir bool
var ImgDirReadAttempts int