DROP database IF EXISTS wardrobe;
CREATE DATABASE wardrobe;
USE wardrobe;

CREATE TABLE wardobe (
	clothingID INT PRIMARY KEY,
	category VARCHAR(50),
    materical VARCHAR(50),
    warmth VARCHAR(50),
    water_resistance VARCHAR(50),
    silhouette VARCHAR(50)
    
);

INSERT INTO `wardobe` VALUES(
	(1, 'hat', 'polyester', 'versitile', 'mild_resistance', 'not_applicable'),
	(2, 'socks', 'wool', 'for_cold_weather', 'not_applicable', 'slim_fit'),
	(3, 'coat', 'other', 'for_cold_weather', 'high_resistance', 'standard_fit'),
	(4, 'shirt', 'silk', 'versatile', 'no_resistance', 'loose_fit'),
    (5, 'sweater', 'polyester', 'versitile', 'mild_resistance', 'loose_fit'),
	(6, 'hoodie', 'wool', 'for_cold_weather', 'not_applicable', 'oversized_fit'),
	(7, 'pants', 'cotton', 'versatile', 'high_resistance', 'slim_fit'),
	(8, 'shorts', 'polyester', 'versatile', 'standard_fit'),
    (9, 'accessory', 'cotton', 'versatile', 'high_resistance', 'standard_fit'),
	(10, 'shoes', 'cotton', 'versatile', 'mild_resistance', 'standard_fit')   
);

 SELECT * FROM wardrobe;