package core

import (
	"../utils"
	"fmt"
)

type CentroidTracker struct {
	// Next unique object ID
	NextObjectId int
	Objects map[int][]float32
	// Frames disappeared
	Disappeared map[int]int
	MaxDisappeared int
}

//set := make(map[int]bool)
type IntSet struct {
	set map[int]bool
}

func (set *IntSet) Add(i int)bool {
	_, found := set.set[i]
	set.set[i] = true
	return !found
}

func (ct *CentroidTracker) Initialise() {
	ct.NextObjectId = 0
}

func (ct *CentroidTracker) Register(c []float32) {
	ct.NextObjectId++
	if ct.Disappeared == nil {
		ct.Disappeared = make(map[int]int)
	}
	if ct.Objects == nil {
		ct.Objects = make(map[int][]float32)
	}
	ct.Objects[ct.NextObjectId] = c
	ct.Disappeared[ct.NextObjectId] = 0
}

func (ct *CentroidTracker) Deregister(id int) {
	delete(ct.Objects, id)
	delete(ct.Disappeared, id)
}

func (ct *CentroidTracker) Update(rects []float32)map[int][]float32 {
	if len(rects) == 0 {
		for k, _ := range ct.Disappeared {
			ct.Disappeared[k]++
			if ct.Disappeared[k] > ct.MaxDisappeared {
				ct.Deregister(k)
			}
		}
		return ct.Objects
	}

	inputCentroids := rects

	if len(ct.Objects) == 0 {
		ct.Register(rects)
	} else {
		var objectIds []int
		var centroids [][]float32
		for k, v := range ct.Objects {
			objectIds = append(objectIds, k)
			centroids = append(centroids, v)
		}
		D := utils.CDist(centroids, inputCentroids)
		fmt.Println(D)
		var usedRows IntSet
		for i, _ := range centroids {
			for _, c := range objectIds {
				if c == i {
					usedRows.Add(i)
				}
			}
		}

	}
	return ct.Objects
}