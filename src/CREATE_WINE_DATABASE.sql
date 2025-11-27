DROP DATABASE IF EXISTS `WineQT`;
CREATE DATABASE `WineQT`;
USE `WineQT`;

CREATE TABLE WineQT.WineData
(`fixed acidity` FLOAT NULL,
`volatile acidity` FLOAT NULL,
`citric acid` FLOAT NULL,
`residual sugar` FLOAT NULL,
`chlorides` FLOAT NULL,
`free sulfur dioxide` FLOAT NULL,
`total sulfur dioxide` FLOAT NULL,
`density` FLOAT NULL,
`pH` FLOAT NULL,
`sulphates` FLOAT NULL,
`alcohol` FLOAT NULL,
`quality` INT NULL,
`id` INT PRIMARY KEY);

LOAD DATA INFILE '/var/lib/mysql-files/WineQT.csv'
INTO TABLE WineData
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 LINES;