// src/main.rs

use actix_web::{web, App, HttpServer, Responder};
use serde::Deserialize;
mod knn;
use knn::KNN;

#[derive(Deserialize)]
struct InputData {
    features: Vec<Option<f64>>,
}

async fn predict(data: web::Json<InputData>, model: web::Data<KNN>) -> impl Responder {
    let prediction = model.predict(data.features.clone());
    format!("Prediction: {}", prediction)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let mut knn = KNN::new(3);
    let model_data = vec![
        (vec![Some(1.0), Some(2.0)], "Class1".to_string()),
        (vec![Some(1.5), Some(1.8)], "Class1".to_string()),
        (vec![Some(5.0), Some(8.0)], "Class2".to_string()),
        (vec![None, Some(2.5)], "Class1".to_string()), // Example with missing data
    ];
    knn.fit(model_data);
    let knn_data = web::Data::new(knn);

    HttpServer::new(move || {
        App::new()
            .app_data(knn_data.clone())
            .route("/predict", web::post().to(predict))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}