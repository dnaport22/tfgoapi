package utils

var maleTags = []string{
	"boy",
	"guy",
	"man",
	"male",
}

var femaleTags = []string{
	"girl",
	"women",
	"female",
}

func MinMax(array []int) (int, int) {
	var max int = array[0]
	var min int = array[0]
	for _, value := range array {
		if max < value {
			max = value
		}
		if min > value {
			min = value
		}
	}
	return min, max
}

func (g *Gender) GetGender()string {
	var female []int
	var male []int
	if len(g.Label) > 0 {
		return g.Label
	}
	for i, v := range g.Assoc {
		if contains(femaleTags, v) {
			female = append(female, i)
		}
	}
	for i, v := range g.Assoc {
		if contains(maleTags, v) {
			male = append(male, i)
		}
	}
	if len(female) > 0 || len(male) > 0 {
		_, mMax := MinMax(male)
		_, fMax := MinMax(female)
		if fMax < mMax {
			return "Female"
		}
		return "Male"
	}
	return "null"
}