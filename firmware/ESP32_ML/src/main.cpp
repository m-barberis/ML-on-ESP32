#include <cstddef>
#include <cstdint>
#include <cstring>

#include "esp_log.h"
#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

extern const uint8_t model_data[] asm("_binary_rps_cnn_tflite_start");
extern const uint8_t image_data[] asm("_binary_test_paper_bin_start");
extern const uint8_t image_end[] asm("_binary_test_paper_bin_end");

namespace {
constexpr char kTag[] = "rps";
constexpr const char* kLabels[] = {"paper", "rock", "scissors"};
constexpr int kClassCount = 3;
constexpr int kExpectedClass = 0;  // Bundled paper fixture.
constexpr int kImageSize = 64;
constexpr int kChannels = 3;
constexpr size_t kImageBytes = kImageSize * kImageSize * kChannels;
alignas(16) uint8_t tensor_arena[112 * 1024];
}  // namespace

extern "C" void app_main() {
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(kTag, "Unsupported model schema");
        return;
    }

    tflite::MicroMutableOpResolver<5> resolver;
    if (resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddMaxPool2D() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk) {
        ESP_LOGE(kTag, "Could not register all model operators");
        return;
    }

    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, sizeof(tensor_arena));
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(kTag, "Tensor allocation failed with %zu bytes", sizeof(tensor_arena));
        return;
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    if (input->type != kTfLiteInt8 || input->bytes != kImageBytes ||
        input->dims->size != 4 || input->dims->data[0] != 1 ||
        input->dims->data[1] != kImageSize || input->dims->data[2] != kImageSize ||
        input->dims->data[3] != kChannels ||
        output->type != kTfLiteInt8 || output->bytes != kClassCount) {
        ESP_LOGE(kTag, "Unexpected model input or output");
        return;
    }

    const auto image_bytes =
        reinterpret_cast<uintptr_t>(image_end) - reinterpret_cast<uintptr_t>(image_data);
    if (image_bytes != kImageBytes) {
        ESP_LOGE(kTag, "Expected %zu bytes of quantized RGB image data", kImageBytes);
        return;
    }
    std::memcpy(input->data.int8, image_data, kImageBytes);

    const int64_t start_us = esp_timer_get_time();
    if (interpreter.Invoke() != kTfLiteOk) {
        ESP_LOGE(kTag, "Inference failed");
        return;
    }
    const double elapsed_ms = (esp_timer_get_time() - start_us) / 1000.0;

    int best = 0;
    for (int i = 0; i < kClassCount; ++i) {
        const int score = output->data.int8[i];
        const float probability = (score - output->params.zero_point) * output->params.scale;
        ESP_LOGI(kTag, "%s = %.3f (int8 %d)", kLabels[i], probability, score);
        if (score > output->data.int8[best]) {
            best = i;
        }
    }

    ESP_LOGI(kTag, "Prediction: %s | inference: %.2f ms | arena: %zu/%zu bytes",
             kLabels[best], elapsed_ms, interpreter.arena_used_bytes(), sizeof(tensor_arena));
    if (best == kExpectedClass) {
        ESP_LOGI(kTag, "PASS: bundled image classified as paper");
    } else {
        ESP_LOGE(kTag, "FAIL: bundled image should classify as paper");
    }
}
