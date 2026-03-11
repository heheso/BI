CREATE DATABASE Sales_Staging;

USE Sales_Staging;

CREATE TABLE Stg_Sales
(
Sale_ID INT,
Product_Name VARCHAR(100),
Customer_Name VARCHAR(100),
Sale_Date DATE,
Quantity INT,
Amount DECIMAL(10,2)
);

CREATE TABLE stg_LegacySales (
    Date VARCHAR(50),
    Region VARCHAR(50),
    Product VARCHAR(50),
    Quantity VARCHAR(50),
    UnitPrice VARCHAR(50),
    StoreLocation VARCHAR(50),
    CustomerType VARCHAR(50),
    Discount VARCHAR(50),
    Salesperson VARCHAR(50),
    TotalPrice VARCHAR(50),
    PaymentMethod VARCHAR(50)
);
