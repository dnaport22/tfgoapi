package utils

var clothingObjects = []string{
	"uniform",
	"hat",
	"suit",
	"glasses",
	"shirt",
	"top",
	"denim",
	"skirt",
	"traditional",
}

var colors = []string{
	"white",
	"black",
	"red",
	"green",
	"blue",
	"yellow",
}

func (c *Clothing) WhichClothing()string  {
	if c.Group.Western > c.Group.Indian {
		return "Western"
	}
	return "Indian"
}

func (c *Clothing) AssocTags()[]string {
	var assoc []string
	for _, v := range c.Assoc {
		if contains(clothingObjects, v) {
			assoc = append(assoc, v)
		}
	}
	return assoc
}