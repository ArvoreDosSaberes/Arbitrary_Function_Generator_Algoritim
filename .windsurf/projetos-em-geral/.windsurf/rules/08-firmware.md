---
<<<<<<< HEAD
trigger: always_on
=======
trigger: model_decision
description: Deve ser aplicada sempre que estiver intervindo na pasta firmware, em especial quando o usuário solicitar alguma alteração no código especifico.
>>>>>>> main
---

## **Microcontrolador**

* Utilize preferencialmente o **ESP32-C6** como plataforma padrão de desenvolvimento.
* O **framework oficial** a ser usado é o **ESP-IDF** (Espressif IoT Development Framework).
* Todas as implementações devem seguir as práticas recomendadas da Espressif, incluindo:

  * Estrutura modular de firmware (componentes independentes);
  * Uso de FreeRTOS nativo do ESP-IDF;
  * Configuração de Wi-Fi e BLE conforme requisitos de projeto;
  * Implementação de logs e monitoramento via `esp_log` com níveis configuráveis (INFO, WARN, ERROR);
  * Otimização de desempenho utilizando `menuconfig` para ajuste de clock, heap e tarefas.