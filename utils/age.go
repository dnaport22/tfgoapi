package utils

func (a *DetectedObject) AgeGroup()string  {
	switch age := a.Age; {
	case age > 0 && age < 4:
		return "0-3"
	case age > 3 && age < 13:
		return "4-12"
	case age > 12 && age < 26:
		return "13-25"
	case age > 25 && age < 46:
		return "26-45"
	case age > 45 && age < 61:
		return "46-60"
	case age > 61:
		return "60+"
	default:
		return "null"
	}
}
