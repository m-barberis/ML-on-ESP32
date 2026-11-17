idf_component_register(
    SRCS "main.cpp"
    PRIV_REQUIRES esp_driver_uart esp_timer
    EMBED_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../../models/int8/final/rps_cnn.tflite"
)
