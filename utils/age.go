package utils

func AgeGroup(a int)string  {
	age := a
	if age > 1 && age <= 3 {
		return "0-3"
	}
	if age > 3 && age <= 12 {
		return "3-12"
	}
	if age > 12 && age <= 25 {
		return "12-25"
	}
	if age > 25 && age <= 45 {
		return "25-45"
	}
	if age > 45 && age <= 60 {
		return "45-60"
	}
	if age > 60 {
		return "60+"
	}
	return "null"
}
