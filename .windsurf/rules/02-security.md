---
trigger: model_decision
description: Quando criando e mantendo código deve sempre ser observado, preservando a segurança do sistema.
---

## **Segurança da Aplicação e Autenticação**

A segurança da aplicação deve ser projetada de forma **estruturada, hierárquica e baseada em criptografia assimétrica**, garantindo a integridade, confidencialidade e autenticidade das comunicações entre cliente e servidor.

---

### **1. Estrutura Criptográfica Inicial**

A autenticação e comunicação entre os componentes do sistema devem iniciar com um **par de chaves pública e privada**, gerado individualmente para cada instância de cliente e servidor.
Esse par estabelece a **base de confiança inicial** para todo o ecossistema, permitindo:

* **Assinatura e verificação** de mensagens;
* **Troca segura de chaves temporárias**;
* **Identificação unívoca** de cada agente na rede.

---

### **2. Integração com Blockchain**

Após o handshake inicial via chaves assimétricas, as trocas subsequentes de chaves e identificadores devem ser **validadas e registradas em blockchain**, criando um **mecanismo de rastreabilidade e imutabilidade** das autenticações.

Cada chave de sessão é derivada a partir do **hash criptográfico de identificação do usuário armazenado na blockchain**, garantindo:

* **Prova de autenticidade descentralizada**;
* **Impossibilidade de falsificação de identidade**;
* **Rotação periódica automática de chaves** baseada em políticas definidas (ex.: tempo, número de transações, ou eventos de login).

Esse modelo permite que a identidade do usuário seja validada sem depender unicamente de servidores centralizados, aumentando a resiliência e auditabilidade do sistema.

---

### **3. Mecanismos de Login e Identificação**

O sistema deve permitir múltiplas formas de autenticação, mantendo **compatibilidade com provedores externos** e **validação interna via blockchain**:

* **Login tradicional:** e-mail e senha (armazenada de forma criptografada, com *salting* e *hashing*).
* **Login federado:** Google, LinkedIn e Facebook (via OAuth 2.0 e OpenID Connect).
* **Login descentralizado:** identificação autenticada por hash único do usuário na blockchain, usado como chave de referência global.

A autenticação federada deve ser integrada ao mesmo mecanismo de registro e verificação criptográfica, garantindo **uniformidade no controle de identidade** independentemente do método de login escolhido.

---

### **4. Rotação e Renovação de Chaves**

As chaves de sessão devem ser **renovadas periodicamente** ou sob condições específicas (como tempo de expiração ou logout).
Cada nova chave é derivada de um **código determinístico** calculado a partir do **hash blockchain do usuário + carimbo temporal + nonce aleatório**, assegurando:

* **Sigilo direto futuro (forward secrecy)**;
* **Proteção contra replay attacks**;
* **Impossibilidade de predição de chaves futuras.**

---

### **5. Boas Práticas Complementares**

* Todas as comunicações devem ocorrer sob **canal TLS 1.3** ou superior.
* As chaves privadas **nunca devem ser transmitidas** — apenas armazenadas localmente e protegidas por hardware seguro (HSM, TPM ou enclave de segurança).
* O sistema deve possuir **mecanismos de revogação** para chaves comprometidas.
* Todos os eventos de autenticação e troca de chaves devem ser **registrados em log estruturado e auditável**.