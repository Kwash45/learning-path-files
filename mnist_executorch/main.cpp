// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: BSD-3-Clause-Clear
/**
 * @file main.c
 * @brief ExecuTorch MNIST Demo for Alif Ensemble E8
 */

#include "RTE_Components.h"
#include CMSIS_device_header

#include "Driver_IO.h"
#include "board_config.h"
#include "pinconf.h"
#include "executorch_runner.h"

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "assets/input_mnist.h"
#include "assets/mnist_model_data.h"

#define mnist_model_data _home_developer_output_mnist_ethos_u55_pte
#define mnist_model_len _home_developer_output_mnist_ethos_u55_pte_len

/* GPIO drivers for DevKit-E8 LEDRGB0. */
extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_LEDRGB0_R_GPIO_PORT);
extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_LEDRGB0_G_GPIO_PORT);
extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_LEDRGB0_B_GPIO_PORT);

static ARM_DRIVER_GPIO *ledR = &ARM_Driver_GPIO_(BOARD_LEDRGB0_R_GPIO_PORT);
static ARM_DRIVER_GPIO *ledG = &ARM_Driver_GPIO_(BOARD_LEDRGB0_G_GPIO_PORT);
static ARM_DRIVER_GPIO *ledB = &ARM_Driver_GPIO_(BOARD_LEDRGB0_B_GPIO_PORT);

static void led_init(void);
static void led_set(int r, int g, int b);
static int enable_sram0_power(void);

static int board_led_pins_config(void)
{
    int32_t status;

    status = pinconf_set(PORT_(BOARD_LEDRGB0_R_GPIO_PORT), BOARD_LEDRGB0_R_GPIO_PIN,
                         BOARD_LEDRGB0_R_ALTERNATE_FUNCTION, PADCTRL_OUTPUT_DRIVE_STRENGTH_2MA);
    if (status != 0) {
        return status;
    }

    status = pinconf_set(PORT_(BOARD_LEDRGB0_G_GPIO_PORT), BOARD_LEDRGB0_G_GPIO_PIN,
                         BOARD_LEDRGB0_G_ALTERNATE_FUNCTION, PADCTRL_OUTPUT_DRIVE_STRENGTH_2MA);
    if (status != 0) {
        return status;
    }

    status = pinconf_set(PORT_(BOARD_LEDRGB0_B_GPIO_PORT), BOARD_LEDRGB0_B_GPIO_PIN,
                         BOARD_LEDRGB0_B_ALTERNATE_FUNCTION, PADCTRL_OUTPUT_DRIVE_STRENGTH_2MA);
    return status;
}

int main(void)
{
    printf("\r\n");
    printf("========================================\r\n");
    printf("  ExecuTorch MNIST NPU Demo\r\n");
    printf("  Alif Ensemble E8 - Cortex-M55 HE\r\n");
    printf("========================================\r\n\r\n");

    led_init();
    led_set(1, 0, 0);

    printf("Initializing SRAM0 power...\r\n");
    if (enable_sram0_power() != 0) {
        printf("ERROR: Failed to enable SRAM0\r\n");
        led_set(1, 0, 0);
        while (1) {
        }
    }
    printf("SRAM0 enabled successfully\r\n\r\n");

    printf("Loading model (%u bytes)...\r\n", (unsigned)mnist_model_len);
    if (executorch_init(mnist_model_data, mnist_model_len) != 0) {
        printf("ERROR: Model initialization failed\r\n");
        led_set(1, 0, 0);
        while (1) {
        }
    }

    led_set(0, 0, 1);

    if (input_mnist_len != 784) {
        printf("ERROR: MNIST input length is %u, expected 784\r\n",
               (unsigned)input_mnist_len);
        led_set(1, 0, 0);
        while (1) {
        }
    }

    printf("Input sample: %u bytes\r\n", (unsigned)input_mnist_len);
    printf("\r\n");

    int8_t output[10];
    memset(output, 0, sizeof(output));

    printf("Running inference...\r\n");
    led_set(0, 1, 0);

    if (executorch_run_inference(input_mnist, input_mnist_len, output, sizeof(output)) != 0) {
        printf("ERROR: Inference failed\r\n");
        led_set(1, 0, 0);
        while (1) {
        }
    }

    int predicted_digit = 0;
    int8_t max_score = output[0];
    for (int i = 1; i < 10; i++) {
        if (output[i] > max_score) {
            max_score = output[i];
            predicted_digit = i;
        }
    }

    printf("\r\nInference completed!\r\n");
    printf("Predicted digit: %d (confidence: %d%%)\r\n",
           predicted_digit, (max_score * 100) / 127);
    printf("\r\nOutput scores:\r\n");
    for (int i = 0; i < 10; i++) {
        printf("  Digit %d: %d\r\n", i, output[i]);
    }

    led_set(0, 1, 0);

    printf("\r\nDemo complete. System halted.\r\n");

    while (1) {
    }
}

static void led_init(void)
{
    int32_t status;

    status = board_led_pins_config();
    if (status != 0) {
        printf("WARNING: LED pin configuration failed: %ld\r\n", (long)status);
        return;
    }

    status = ledR->Initialize(BOARD_LEDRGB0_R_GPIO_PIN, NULL);
    status |= ledG->Initialize(BOARD_LEDRGB0_G_GPIO_PIN, NULL);
    status |= ledB->Initialize(BOARD_LEDRGB0_B_GPIO_PIN, NULL);
    if (status != ARM_DRIVER_OK) {
        printf("WARNING: LED GPIO initialization failed\r\n");
        return;
    }

    status = ledR->PowerControl(BOARD_LEDRGB0_R_GPIO_PIN, ARM_POWER_FULL);
    status |= ledG->PowerControl(BOARD_LEDRGB0_G_GPIO_PIN, ARM_POWER_FULL);
    status |= ledB->PowerControl(BOARD_LEDRGB0_B_GPIO_PIN, ARM_POWER_FULL);
    if (status != ARM_DRIVER_OK) {
        printf("WARNING: LED GPIO power control failed\r\n");
        return;
    }

    status = ledR->SetDirection(BOARD_LEDRGB0_R_GPIO_PIN, GPIO_PIN_DIRECTION_OUTPUT);
    status |= ledG->SetDirection(BOARD_LEDRGB0_G_GPIO_PIN, GPIO_PIN_DIRECTION_OUTPUT);
    status |= ledB->SetDirection(BOARD_LEDRGB0_B_GPIO_PIN, GPIO_PIN_DIRECTION_OUTPUT);
    if (status != ARM_DRIVER_OK) {
        printf("WARNING: LED GPIO direction setup failed\r\n");
        return;
    }

    led_set(0, 0, 0);
}

static void led_set(int r, int g, int b)
{
    (void)ledR->SetValue(BOARD_LEDRGB0_R_GPIO_PIN,
                         r ? GPIO_PIN_OUTPUT_STATE_HIGH : GPIO_PIN_OUTPUT_STATE_LOW);
    (void)ledG->SetValue(BOARD_LEDRGB0_G_GPIO_PIN,
                         g ? GPIO_PIN_OUTPUT_STATE_HIGH : GPIO_PIN_OUTPUT_STATE_LOW);
    (void)ledB->SetValue(BOARD_LEDRGB0_B_GPIO_PIN,
                         b ? GPIO_PIN_OUTPUT_STATE_HIGH : GPIO_PIN_OUTPUT_STATE_LOW);
}

static int enable_sram0_power(void)
{
    /*
     * The tutorial's SERVICES_power_request/SERVICES_power_sram_t API is not
     * present in AlifSemiconductor.Ensemble 2.0.4. Keep this as a no-op until
     * the pack-specific SRAM power call is added.
     */
    return 0;
}
