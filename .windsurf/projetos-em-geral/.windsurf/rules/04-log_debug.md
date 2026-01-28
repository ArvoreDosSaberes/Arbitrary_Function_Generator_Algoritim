---
trigger: model_decision
description: Aplicar quando estiver lidando com saidas de Logs, padronizando o logger do sistema, não importa a linguagem.
---

## **Depuração e Log**

A depuração é uma parte essencial do desenvolvimento de sistemas embarcados e aplicações em Node.js. Todas as mensagens de log devem ser **claras, informativas, didáticas e agrupadas por contexto**, de forma a facilitar o diagnóstico e o rastreamento de eventos em tempo real.

---

### **Padrões Gerais de Saída**

* As mensagens de log devem seguir o formato:

  ```
  [setor/tema] Mensagem informativa
  ```

  **Exemplo:**

  ```
  [auth/login] ✅ Token JWT gerado com sucesso.
  [db/connection] ⚙️ Reconnecting to MySQL...
  [sensor/adc] 📊 Sampling rate: 10kHz, mean value: 512
  ```

* Utilize **emoticons** para destacar estados e eventos importantes:

  * ✅ Sucesso
  * ⚠️ Alerta
  * ❌ Erro
  * 🧠 Informação técnica
  * 📊 Estatísticas ou medições

* Sempre exponha **estatísticas relevantes** do sistema, como:

  * Tempo de execução (ms)
  * Consumo de memória ou CPU
  * Número de requisições atendidas
  * Pacotes processados ou bytes transmitidos

---

### **Node.js – Logger Padronizado**

Em aplicações Node.js, utilize o **Pino** como sistema padrão de logging.

* O logger deve possuir:

  * **Níveis hierárquicos de log** (`trace`, `debug`, `info`, `warn`, `error`, `fatal`);
  * **Saída estruturada em JSON**, compatível com ferramentas como **Grafana Loki**, **ELK** ou **Fluentd**;
  * **Rotação e compressão** de arquivos de log, quando aplicável.

**Exemplo mínimo de configuração:**

```javascript
import pino from 'pino';

const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  transport: {
    target: 'pino-pretty',
    options: { colorize: true, translateTime: 'SYS:standard' }
  }
});

logger.info('[init/system] ✅ Serviço iniciado com sucesso');
logger.error('[db/connection] ❌ Falha ao conectar ao banco');
```

---

### **Compatibilidade com o Baileys**

Ao integrar sistemas com o **Baileys**, use o **logger nativo** do pacote para manter o **mesmo layout, estilo e formato de saída**.

* Importe diretamente de `@whiskeysockets/baileys`;
* Preserve o formato padrão: **nível + timestamp + contexto**;
* Evite sobrescrever ou duplicar o logger existente;
* Redirecione logs personalizados para o mesmo fluxo, mantendo coloração e estrutura.

**Exemplo:**

```javascript
import makeWASocket, { useMultiFileAuthState, fetchLatestBaileysVersion, logger } from '@whiskeysockets/baileys';

logger.info('[baileys/init] 🧠 Inicializando cliente WhatsApp...');
```

---

### **Arquitetura de Logging**

* Implemente uma **classe de abstração** para o sistema de logs, permitindo trocar facilmente o framework utilizado (ex.: Pino, Winston, ou customizado).
* Utilize um **factory** para criação do logger, identificando a origem das mensagens no formato:

  ```
  [origem] restante da mensagem
  ```

---

### **Firmware e Sistemas Embarcados**

Nos sistemas embarcados, os logs devem ser enviados pela **serial UART**, mantendo formatação semelhante aos padrões gerais:

```
[sensor/temp] 📈 Temperature: 36.4°C
[wifi/status] 🌐 Connected to SSID: LabNet
[task/adc_dma] ⚙️ DMA Buffer Overrun Detected
```

* Utilize **macros padronizadas** (`LOG_INFO`, `LOG_WARN`, `LOG_ERROR`) para consistência.
* Em ambientes com **RTOS** (ex.: FreeRTOS), utilize **mecanismos task-safe** (mutexes, buffers circulares) para evitar interferência entre tarefas.

---

### **Boas Práticas de Depuração**

* Mantenha mensagens de log **sintéticas e úteis**, evitando ruído excessivo.
* Diferencie logs temporários de depuração (`debug`) dos logs permanentes de operação (`info`, `warn`, `error`).
* Nas fases de produção, reduza a verbosidade e direcione os logs a sistemas de coleta ou armazenamento persistente.