---
trigger: model_decision
description: Deve ser aplicada sempre que o usuário solicitar a criação de um script.
---

## **Scripts**

* Todos os **scripts de automação, manutenção e suporte** devem ser armazenados na pasta:

  ```
  /scripts/
  ```

  localizada na raiz do projeto.
* Essa organização facilita a manutenção, versionamento e execução automatizada em pipelines CI/CD.
* Cada script deve conter **cabeçalhos de documentação** descrevendo seu propósito, autor, data e dependências necessárias.

---

## **Scripts de Apoio**

* Sempre que possível, **utilize Bash** (`.sh`) para scripts de automação, inicialização e manutenção, garantindo compatibilidade com ambientes Linux e containers.
* Quando o uso de Bash não for viável (por limitações de lógica ou bibliotecas), utilize **Node.js com JavaScript**, priorizando a portabilidade e a integração com o restante do sistema.
* Em casos de **processamento intensivo de dados, imagens ou análise científica**, prefira **Python**.

  * Ao usar Python:

    1. Crie um ambiente virtual na raiz do projeto, nomeado conforme o padrão:

       ```
       /venv-nomedoprojeto/
       ```
    2. Ative o ambiente virtual antes de instalar as dependências.
    3. Registre todas as dependências no arquivo `requirements.txt` dentro da pasta do projeto.
    4. Documente no cabeçalho do script os comandos necessários para ativação do ambiente e execução do script.