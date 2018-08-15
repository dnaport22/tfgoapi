package utils

func (c *AzureClothing) WhichClothing()string  {
	if c.Western > c.Indian {
		return "Western"
	}
	return "Indian"
}