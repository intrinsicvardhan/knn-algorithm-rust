# Rust KNN Classifier with Web API

This project implements a K-Nearest Neighbors (KNN) classifier in Rust, exposed through a web API using the Actix web framework. The KNN algorithm is capable of handling missing data and provides predictions based on the trained model.

## Features

- KNN classification algorithm implementation in Rust
- Web API for making predictions
- Handling of missing data in both training and prediction
- Simple and efficient Euclidean distance calculation
- Easy-to-use interface for training and prediction

## Project Structure

The project consists of two main Rust files:

1. `src/main.rs`: Contains the web server setup and API endpoint.
2. `src/knn.rs`: Implements the KNN algorithm and related functionality.

## How It Works

1. The KNN model is initialized with a specified number of neighbors (k).
2. Training data is provided to the model, which may include missing values (represented as `None`).
3. The model handles missing data by calculating mean values for each feature from the available data.
4. When a prediction is requested, the model:
   - Handles any missing data in the input
   - Calculates distances to all training points
   - Finds the k nearest neighbors
   - Returns the most common class among these neighbors

## API Usage

The API exposes a single endpoint for predictions:

- **POST** `/predict`
  - Input: JSON object with a `features` array containing numbers or null values
  - Output: Predicted class as a string

Example request:
```json
{
  "features": [1.0, 2.0, null, 3.5]
}