#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "driver/uart.h"
#include "esp_heap_caps.h"
#include "esp_rom_crc.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "sdkconfig.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

extern const uint8_t model_data[] asm("_binary_rps_cnn_tflite_start");
extern const uint8_t model_end[] asm("_binary_rps_cnn_tflite_end");

namespace {
constexpr uart_port_t kUart = static_cast<uart_port_t>(CONFIG_ESP_CONSOLE_UART_NUM);
constexpr uint32_t kHeapCaps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
constexpr int kImageSize = 64;
constexpr int kChannels = 3;
constexpr int kClassCount = 3;
constexpr size_t kImageBytes = kImageSize * kImageSize * kChannels;
alignas(16) uint8_t tensor_arena[112 * 1024];

bool read_image(uint8_t* pixels) {
    for (size_t received = 0; received < kImageBytes;) {
        const int count = uart_read_bytes(kUart, pixels + received, kImageBytes - received,
                                          pdMS_TO_TICKS(2000));
        if (count <= 0) {
            uart_flush_input(kUart);
            return false;
        }
        received += count;
    }
    return true;
}
}  // namespace

extern "C" void app_main() {
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        std::puts("RPS ERROR unsupported_model_schema");
        return;
    }

    static tflite::MicroMutableOpResolver<5> resolver;
    if (resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddMaxPool2D() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk) {
        std::puts("RPS ERROR operator_registration");
        return;
    }

    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, sizeof(tensor_arena));
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        std::puts("RPS ERROR tensor_allocation");
        return;
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    if (input->type != kTfLiteInt8 || input->bytes != kImageBytes ||
        input->dims->size != 4 || input->dims->data[0] != 1 ||
        input->dims->data[1] != kImageSize || input->dims->data[2] != kImageSize ||
        input->dims->data[3] != kChannels ||
        output->type != kTfLiteInt8 || output->bytes != kClassCount) {
        std::puts("RPS ERROR model_input_or_output");
        return;
    }

    // Use the console's configured baud rate and pins, with buffered binary RX.
    if (uart_driver_install(kUart, 2048, 0, 0, nullptr, 0) != ESP_OK) {
        std::puts("RPS ERROR uart_setup");
        return;
    }
    const auto model_bytes =
        reinterpret_cast<uintptr_t>(model_end) - reinterpret_cast<uintptr_t>(model_data);
    const uint32_t model_crc = esp_rom_crc32_le(0, model_data, model_bytes);
    const size_t heap_total = heap_caps_get_total_size(kHeapCaps);

    while (true) {
        uint8_t command;
        if (uart_read_bytes(kUart, &command, 1, portMAX_DELAY) != 1) {
            continue;
        }
        if (command == '?') {
            std::printf("RPS READY 1 %08" PRIx32 " %zu %zu %d\n",
                        model_crc, sizeof(tensor_arena), heap_total, CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
        } else if (command == 'I') {
            auto* pixels = reinterpret_cast<uint8_t*>(input->data.int8);
            if (!read_image(pixels)) {
                std::puts("RPS ERROR incomplete_image");
            } else {
                const uint32_t image_crc = esp_rom_crc32_le(0, pixels, kImageBytes);
                if (heap_caps_monitor_local_minimum_free_size_start() != ESP_OK) {
                    std::puts("RPS ERROR heap_monitor");
                    std::fflush(stdout);
                    continue;
                }
                const int64_t started = esp_timer_get_time();
                const TfLiteStatus status = interpreter.Invoke();
                const int64_t elapsed_us = esp_timer_get_time() - started;
                const size_t heap_after = heap_total - heap_caps_get_free_size(kHeapCaps);
                const size_t heap_peak = heap_total - heap_caps_get_minimum_free_size(kHeapCaps);
                heap_caps_monitor_local_minimum_free_size_stop();

                if (status != kTfLiteOk) {
                    std::puts("RPS ERROR inference");
                } else {
                    std::printf("RPS RESULT %08" PRIx32 " %d %d %d %" PRId64 " %zu %zu %zu\n",
                                image_crc, output->data.int8[0], output->data.int8[1],
                                output->data.int8[2], elapsed_us, interpreter.arena_used_bytes(),
                                heap_after, heap_peak);
                }
            }
        }
        std::fflush(stdout);
    }
}
